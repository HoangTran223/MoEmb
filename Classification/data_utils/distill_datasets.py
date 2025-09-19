import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from Classification.utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer

class DistillDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        student_tokenizer: AutoTokenizer,
        teacher_tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = args.max_length

        self.dataset = self._load_and_process_data()

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.csv")

        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'text' not in df.columns:
                raise ValueError(f"CSV file {path} must contain a 'text' column")
            label_col = 'label' if 'label' in df.columns else 'labels'
            
            log_rank("Processing dataset for classification...")  
            
            # Determine rank safely (works even if torch.distributed is not initialized)
            try:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            except Exception:
                rank = 0
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(rank != 0)):
                student_input_ids = self.student_tokenizer.encode(
                    row['text'], 
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                # Always include the original text in the sample for KD overlap logic
                tokenized_data = {
                    "student_input_ids": student_input_ids,
                    "label": int(row[label_col]),
                    "text": row['text']
                }
                if self.teacher_tokenizer:
                    teacher_input_ids = self.teacher_tokenizer.encode(
                        row['text'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    tokenized_data["teacher_input_ids"] = teacher_input_ids

                    # Precompute span-overlap pairs once to avoid runtime tokenization cost
                    try:
                        t_enc = self.teacher_tokenizer(
                            row['text'],
                            return_offsets_mapping=True,
                            truncation=True,
                            max_length=self.max_length,
                            add_special_tokens=True,
                            return_tensors=None,
                        )
                        s_enc = self.student_tokenizer(
                            row['text'],
                            return_offsets_mapping=True,
                            truncation=True,
                            max_length=self.max_length,
                            add_special_tokens=True,
                            return_tensors=None,
                        )
                        # Normalize shapes (single string vs batched)
                        def _flat_ids(v):
                            if isinstance(v, list) and v and isinstance(v[0], (list, tuple, np.ndarray)):
                                return v[0]
                            return v
                        def _flat_off(v):
                            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)) and len(v[0]) > 0 and isinstance(v[0][0], (list, tuple)):
                                return v[0]
                            return v
                        t_ids_tok = _flat_ids(t_enc["input_ids"])
                        s_ids_tok = _flat_ids(s_enc["input_ids"])
                        t_off = _flat_off(t_enc["offset_mapping"])
                        s_off = _flat_off(s_enc["offset_mapping"])

                        # Compute valid positions (exclude specials/pad)
                        t_pad = getattr(self.teacher_tokenizer, 'pad_token_id', None)
                        s_pad = getattr(self.student_tokenizer, 'pad_token_id', None)
                        t_spec = set(getattr(self.teacher_tokenizer, 'all_special_ids', []) or [])
                        s_spec = set(getattr(self.student_tokenizer, 'all_special_ids', []) or [])
                        def _valid(ids, offs, pad, specs):
                            pos = []
                            for i, (tid, (a, b)) in enumerate(zip(ids, offs)):
                                if (pad is not None and tid == pad) or (tid in specs):
                                    continue
                                if a == 0 and b == 0 and ((pad is not None and tid == pad) or tid in specs):
                                    continue
                                if b <= a:
                                    continue
                                pos.append(i)
                            return pos
                        t_valid = _valid(t_ids_tok, t_off, t_pad, t_spec)
                        s_valid = _valid(s_ids_tok, s_off, s_pad, s_spec)

                        pairs = []
                        for sj in s_valid:
                            sa, sb = s_off[sj]
                            if sb <= sa:
                                continue
                            for ti in t_valid:
                                ta, tb = t_off[ti]
                                if tb <= ta:
                                    continue
                                if (sa < tb) and (ta < sb):
                                    pairs.append((ti, sj))
                        # Store pairs; empty list is acceptable (we'll fallback to runtime recompute if needed)
                        tokenized_data["overlap_pairs"] = pairs
                    except Exception:
                        # Skip precomp on any tokenizer issue; FKD_H will recompute at runtime
                        pass
                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_classification(
        self, i, samp, model_data, no_model_data
    ):
        input_ids = np.array(samp["student_input_ids"])
        input_len = len(input_ids)
        
        model_data["input_ids"][i][:input_len] = torch.tensor(input_ids, dtype=torch.long)
        model_data["attention_mask"][i][:input_len] = 1.0
        no_model_data["labels"][i] = torch.tensor(samp["label"], dtype=torch.long)

        if "teacher_input_ids" in samp:
            t_input_ids = np.array(samp["teacher_input_ids"])
            t_input_len = len(t_input_ids)
            model_data["teacher_input_ids"][i][:t_input_len] = torch.tensor(t_input_ids, dtype=torch.long)
            model_data["teacher_attention_mask"][i][:t_input_len] = 1.0

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        student_pad_token_id = self.student_tokenizer.pad_token_id
        if student_pad_token_id is None:
            student_pad_token_id = 0
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * student_pad_token_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        output_data = {
            "labels": torch.zeros(bs, dtype=torch.long)
        }

        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id
            if teacher_pad_token_id is None:
                teacher_pad_token_id = 0
            model_data.update({
                "teacher_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_attention_mask": torch.zeros(bs, max_length),
            })

        # Also collect the original text fields for each sample (required)
        texts = [samp["text"] if "text" in samp else None for samp in samples]
        overlaps_list = []
        for i, samp in enumerate(samples):
            self._process_classification(i, samp, model_data, output_data)
            # collect precomputed overlaps if any
            overlaps_list.append(samp.get("overlap_pairs", None))
        # Add explicit text fields for FKD_H span-overlap logic (teacher/student can be the same source text)
        model_data["text"] = texts
        model_data["teacher_texts"] = texts
        model_data["student_texts"] = texts
        model_data["overlaps"] = overlaps_list
        return model_data, output_data
