import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,  
    AutoModelForSequenceClassification,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from STS.utils import log_rank
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
            self.teacher_model, self.teacher_tokenizer = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizer = None, None
        # record teacher vocab size explicitly (always use full vocab size)
        try:
            if self.teacher_tokenizer is not None:
                tokenizer_obj = self.teacher_tokenizer
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
        # Pre-create W_q for FKD_A/FKD_H so optimizer captures its params
        if getattr(self.args, 'criterion', None) in ['fkd_a', 'fkd_h'] and self.teacher_model is not None:
            in_dim = getattr(self, 'teacher_hidden_size', None) or getattr(self, 'hidden_size', None)
            out_dim = getattr(self, 'hidden_size', None) or in_dim
            if 'W_q' not in self.projectors and in_dim is not None and out_dim is not None:
                self.projectors['W_q'] = nn.Linear(in_dim, out_dim)
    # FKD uses projectors only (t2s recommended). No EAADP modules.

    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tokenizer
        
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
        # Load the backbone model using AutoModel for STS
        backbone = AutoModel.from_pretrained(self.args.model_path, trust_remote_code=True)
        hidden_size = backbone.config.hidden_size
        # For STS tasks, we use a regression head (num_labels=1) or classification head as per args
        num_labels = 1
        model = SequenceClassifierWrapper(backbone, hidden_size, num_labels)
        # Integrate supervised adapter if PEFT is specified (removing simcse modifications)
        if self.args.peft is not None and hasattr(self.args, 'peft_config'):
            model = get_peft_model(model, self.args.peft_config)
        if self.args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        return model, self.load_tokenizer(self.args.model_path)
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(
            self.args.teacher_model_path,
            trust_remote_code=True
        )
        backbone = AutoModel.from_pretrained(self.args.teacher_model_path, config=config, trust_remote_code=True)
        hidden_size = backbone.config.hidden_size
        num_labels = self.args.num_labels  # Use provided number of labels
        teacher_model = SequenceClassifierWrapper(backbone, hidden_size, num_labels)
        # (Optional) Apply supervised adapter merging if required
        return teacher_model, self.load_tokenizer(self.args.teacher_model_path)

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

# New wrapper for STS tasks using supervised adapter
class SequenceClassifierWrapper(nn.Module):
    def __init__(self, backbone, hidden_size, num_labels):
        super(SequenceClassifierWrapper, self).__init__()
        self.backbone = backbone
        # For STS regression tasks, num_labels is typically 1
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, *args, **kwargs):
        outputs = self.backbone(*args, **kwargs)
        # Support both tuple and ModelOutput
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs
        # For STS, use the [CLS] token representation (first token) as pooled output
        pooled = last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits
