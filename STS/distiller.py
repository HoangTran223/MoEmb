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
        
        if self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizers = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizers = None, {}
        if self.teacher_model and self.args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")
        # Pre-create W_q for FKD_A so optimizer captures its params
        if getattr(self.args, 'criterion', None) in ['fkd_a'] and self.teacher_model is not None:
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
        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError(f"Invalid model_dtype for {self.args.model_dtype}")

        # Always force regression for STS: num_labels=1
        num_labels = 1

        if self.args.peft is not None:  # for LLM2Vec student
            if self.args.peft == "lora":
                config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True)
                config.is_model_parallel = False
                config.num_labels = num_labels
                tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
                if hasattr(config, "n_embed"):
                    self.hidden_size = config.n_embed
                else:
                    self.hidden_size = config.hidden_size
                model = AutoModelForSequenceClassification.from_pretrained(
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                model.config.pad_token_id = 2
                # merge pretrained lora adapters (MNTP and simcse)
                model = PeftModel.from_pretrained(
                    model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                ).merge_and_unload()
                model = PeftModel.from_pretrained(
                    model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
                ).merge_and_unload()
                # apply new lora for fine-tuning if training
                if self.args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"
                        ]
                    )
                    model = get_peft_model(model, peft_config)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in model.parameters())
                    print(f"Trainable parameters: {trainable_params}/{all_params} ({trainable_params/all_params:.2%})")
            else:
                raise NotImplementedError
        else:  # for BERT
            config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
            config.is_model_parallel = False
            config.num_labels = num_labels
            tokenizer = self.load_tokenizer("bert-base-uncased")
            if hasattr(config, "n_embed"):
                self.hidden_size = config.n_embed
            else:
                self.hidden_size = config.hidden_size
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                config=config,
                device_map=None,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        # If the classifier head is wrong shape, re-initialize for regression
        if hasattr(model, 'classifier'):
            out_features = model.classifier.out_features if hasattr(model.classifier, 'out_features') else model.classifier.out_features
            if out_features != 1:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, 1).to(model.device)

        if self.args.gradient_checkpointing:
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

        config.num_labels = self.args.num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        model.config.pad_token_id = 2
        teacher_model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        )    
        
        teacher_model = teacher_model.merge_and_unload()  # This can take several minutes on cpu

        # Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
        teacher_model = PeftModel.from_pretrained(
            teacher_model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
        )
        teacher_model = teacher_model.merge_and_unload()

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

            # Optional classifier head loading
            classifier_path = os.path.join(adapter_dir, "classifier_head.bin")
            if os.path.exists(classifier_path):
                try:
                    log_rank("Loading classifier head from trained model...")
                    # Use weights_only=True (PyTorch>=2.6) and fallback if needed
                    try:
                        classifier_state_dict = torch.load(classifier_path, map_location="cpu", weights_only=True)
                    except TypeError:
                        classifier_state_dict = torch.load(classifier_path, map_location="cpu")
                    teacher_model.score.load_state_dict(classifier_state_dict)
                except Exception as e:
                    log_rank(f"[Teacher][WARN] Failed to load classifier head: {e}")
            else:
                log_rank("No classifier head found in teacher model path. Using default classifier.")
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
