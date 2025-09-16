import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from Classification.utils import log_rank
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN", None)
if hf_token:
    try:
        login(token=hf_token)
    except Exception:
        pass


class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        # ensure a container for optional projectors exists early
        self.projectors = nn.ModuleDict()
        self.student_model, self.student_tokenizer = self.load_student_model()
        # record student vocab size explicitly (always use full vocab size)
        try:
            self.student_vocab_size = getattr(self.student_tokenizer, 'vocab_size', None)
            if self.student_vocab_size is None and self.student_tokenizer is not None:
                self.student_vocab_size = len(self.student_tokenizer)
        except Exception:
            self.student_vocab_size = None
        if self.student_vocab_size is not None:
            log_rank(f"[Distiller] Student vocab size: {self.student_vocab_size}")
        
        if self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizers = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizers = None, {}
        # record teacher vocab size explicitly (always use full vocab size)
        try:
            if self.teacher_tokenizers is not None and self.teacher_tokenizers != {}:
                tokenizer_obj = self.teacher_tokenizers
                self.teacher_vocab_size = getattr(tokenizer_obj, 'vocab_size', None)
                if self.teacher_vocab_size is None:
                    self.teacher_vocab_size = len(tokenizer_obj)
            else:
                self.teacher_vocab_size = None
        except Exception:
            self.teacher_vocab_size = None
        if self.teacher_vocab_size is not None:
            log_rank(f"[Distiller] Teacher vocab size: {self.teacher_vocab_size}")
        if self.teacher_model and self.args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")
        # Pre-create W_q for FKD_A/FKD_H so optimizer captures its params (if t2s not provided)
        if getattr(self.args, 'criterion', None) in ['fkd_a', 'fkd_h'] and self.teacher_model is not None:
            in_dim = getattr(self, 'teacher_hidden_size', None) or getattr(self, 'hidden_size', None)
            out_dim = getattr(self, 'hidden_size', None) or in_dim
            if ('W_q' not in self.projectors) and (in_dim is not None) and (out_dim is not None):
                self.projectors['W_q'] = nn.Linear(in_dim, out_dim)
    # FKD uses projectors only (t2s recommended). No EAADP modules.

    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tokenizer

    class SequenceClassifierWrapper(nn.Module):
        """Wrap an AutoModel with a simple classification head.
        - For encoder models (e.g., BERT): use CLS token (index 0)
        - For decoder-only models (e.g., Mistral): use last token (index -1)
        Returns SequenceClassifierOutput with logits, loss (optional) and hidden_states.
        """
        def __init__(self, base_model: AutoModel, hidden_size: int, num_labels: int, dtype: torch.dtype):
            super().__init__()
            self.base_model = base_model
            self.config = getattr(base_model, 'config', None)
            self.hidden_size = hidden_size
            self.num_labels = num_labels
            self.classifier = nn.Linear(hidden_size, num_labels, dtype=dtype, device=next(base_model.parameters()).device)
            # naming compatibility for downstream save logic
            # BERT-like
            self.classifier_name = 'classifier'

        def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True, **kwargs):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state
            # decide pooling by config
            use_last_token = bool(getattr(self.config, 'is_decoder', False))
            if use_last_token:
                pooled = last_hidden[:, -1, :]
            else:
                pooled = last_hidden[:, 0, :]
            logits = self.classifier(pooled)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            if return_dict:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=getattr(outputs, 'attentions', None),
                )
            return logits
        
    def set_and_load_existing_projectors(self):
        self.projectors = nn.ModuleDict()
        projector_config = json.load(open(self.args.projector_config_path))
        name_dict = {
            "s": self.hidden_size, 
            "t": self.teacher_hidden_size,
            "relu": nn.ReLU()
        }
        # auto-parse projector config strings to construct nn.Module
        for projector_name in projector_config:
            # for d in projector_config[loc]:
            if projector_config[projector_name]["enabled"]:
                self.projectors[projector_name] = nn.Sequential()

                structure = projector_config[projector_name]["structure"].split("-")
                for i in range(len(structure)):
                    if structure[i] not in ["relu"]:
                        coef = 1 if not len(structure[i][:-1]) else int(structure[i][:-1])
                        base_size = name_dict[structure[i][-1]]
                        structure[i] = coef * base_size

                for i in range(len(structure) - 1):
                    if isinstance(structure[i], int) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(structure[i], structure[i+1])
                        )
                    elif isinstance(structure[i], int) and isinstance(structure[i+1], str):
                        self.projectors[projector_name].append(
                            name_dict[structure[i+1]]
                        )
                        last_size = structure[i]
                    elif isinstance(structure[i], str) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(last_size, structure[i+1])
                        )
                    else:
                        raise NotImplementedError(f"Invalid structure for '{structure}'")
                        
        # load existing projectors if already have
        self.load_existing_projectors()

    def load_existing_projectors(self):
        if self.args.projector_path is not None:
            projector_path = os.path.join(self.args.projector_path, "projector.pt")
        else:
            projector_path = os.path.join(self.args.model_path, "projector.pt")

        if os.path.exists(projector_path):
            projector_params = torch.load(projector_path, map_location=f"cuda:{self.device}")
            log_rank("Existing projector params: {}".format(list(projector_params.keys())))
            for key in self.projectors:
                try:
                    state_dict = {
                        n.split('.', 1)[1]: projector_params[n] for n in projector_params if n.startswith(key)
                    }
                    self.projectors[key].load_state_dict(state_dict)
                    log_rank("Load projector '{}' from current path.".format(key))
                except:
                    log_rank("Not compatible for projector '{}'".format(key))
                    continue
    
    def load_student_model(self):
        log_rank("Loading student model...")
    
        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError("Invalid model_dtype for f`{self.args.model_dtype}`")

        if self.args.peft is not None: #for LLM2Vec student
            if self.args.peft == "lora":
                config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True)
                config.is_model_parallel = False
                tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
                self.hidden_size = getattr(config, "n_embed", getattr(config, "hidden_size", None))
                base = AutoModel.from_pretrained(
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                # merge pretrained lora adapters (MNTP and supervised)
                base = PeftModel.from_pretrained(
                    base, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                ).merge_and_unload()
                base = PeftModel.from_pretrained(
                    base, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised"
                ).merge_and_unload()
                base.config.pad_token_id = getattr(base.config, 'pad_token_id', 2)
                # apply new lora for fine-tuning if training (on base/backbone only)
                if self.args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"
                        ]
                    )
                    base = get_peft_model(base, peft_config)
                    trainable_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in base.parameters())
                    print(f"Trainable parameters (backbone LoRA): {trainable_params}/{all_params} ({trainable_params/all_params:.2%})")
                # wrap with classifier head
                model = Distiller.SequenceClassifierWrapper(base, self.hidden_size, self.args.num_labels, self.dtype)
            else:
                raise NotImplementedError
        else: #for BERT
            config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
            config.is_model_parallel = False
    
            # lấy tokenizer
            tokenizer = self.load_tokenizer("bert-base-uncased")
            
            self.hidden_size = getattr(config, "n_embed", getattr(config, "hidden_size", None))
            base = AutoModel.from_pretrained(
                "bert-base-uncased",
                config=config,
                device_map=None,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            model = Distiller.SequenceClassifierWrapper(base, self.hidden_size, self.args.num_labels, self.dtype)
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        if self.args.gradient_checkpointing:
            # enable on base model if available
            try:
                model.base_model.gradient_checkpointing_enable()
            except Exception:
                model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            trust_remote_code=True
        )
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        # Teacher: use AutoModel backbone and merge MNTP + supervised adapters for feature extraction
        base = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        base.config.pad_token_id = getattr(base.config, 'pad_token_id', 2)
        teacher_model = PeftModel.from_pretrained(
            base,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        )
        teacher_model = teacher_model.merge_and_unload()  # may take time on CPU
        # Load supervised adapter instead of unsupervised simcse
        teacher_model = PeftModel.from_pretrained(
            teacher_model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised"
        ).merge_and_unload()

        if getattr(self.args, 'teacher_model_path', None):
            adapter_dir = self.args.teacher_model_path
            cfg_file = os.path.join(adapter_dir, "adapter_config.json")
            weight_exists = any(os.path.exists(os.path.join(adapter_dir, f)) for f in ["adapter_model.safetensors", "adapter_model.bin"])
            if os.path.isdir(adapter_dir) and os.path.exists(cfg_file) and weight_exists:
                try:
                    log_rank(f"[Teacher] Loading extra local adapter directory: {adapter_dir}")
                    teacher_model = PeftModel.from_pretrained(teacher_model, adapter_dir)
                    teacher_model = teacher_model.merge_and_unload()
                    log_rank("[Teacher] Extra adapter merged.")
                except Exception as e:
                    log_rank(f"[Teacher][WARN] Failed to load extra adapter at {adapter_dir}: {e}")
            else:
                log_rank(f"[Teacher][WARN] Provided teacher_model_path '{adapter_dir}' is not a valid PEFT adapter directory (missing adapter_config.json or weights). Skipping.")

            # No classifier head needed for teacher backbone in FKD; skip loading any classifier
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        return teacher_model, tokenizer
    
    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors") and len(self.projectors) > 0:
            if self.args.projector_lr:
                pretrained_proj = self.args.pretrained_projector.split(",") if self.args.pretrained_projector is not None else []
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b not in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.projector_lr
                })
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.pretrained_projector_lr
                })
            else:
                optimizer.add_param_group({
                    "params": [p for b in self.projectors for p in self.projectors[b].parameters()],
                })
        return optimizer

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch["output_batch"]
        return criterion(self, input_data, output_data, logging_output, loss_denom)
