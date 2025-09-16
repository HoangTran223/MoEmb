# !pip install -q flash-attn --no-build-isolation
import transformers
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from torch import nn


per_device_train_batch_size = 4


# os.environ["KAGGLE_USERNAME"] = "winddao"
# os.environ["KAGGLE_KEY"] = "bf13807bde7b37c0d65d90d8688a9744"
# import kagglehub
# import sys
# kagglehub.login()
# kaggle_api_path = kagglehub.dataset_download('winddao/kaggle-api')
# data_path = kagglehub.dataset_download('winddao/embedding-data')
# module_path = kagglehub.dataset_download('winddao/distillation')
# mistral_path = kagglehub.dataset_download('winddao/mistral-transformer-4-44-2')
#
# print('Data source import complete.')
# dataset_path = kagglehub.dataset_download('sssonnn/banking77')


# with open(mistral_path + '/modeling_mistral.py', "r") as f:
#     modeling_mistral = f.read()
#
# with open(mistral_path + '/configuration_mistral.py', "r") as f:
#     configuration_mistral = f.read()
#
#
# transformers_path_dir = os.path.dirname(transformers.__file__)
# model_path = os.path.join(transformers_path_dir, "models/mistral/modeling_mistral.py")
# config_path = os.path.join(transformers_path_dir, "models/mistral/configuration_mistral.py")
#
# with open(model_path, "w") as f:
#     f.write(modeling_mistral)
#
# with open(config_path, "w") as f:
#     f.write(configuration_mistral)


# sys.path.append(module_path)
from huggingface_hub import login
HF_TOKEN = True

class LLM2VecForSequenceClassification(nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super().__init__()
        model_name = 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp'
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",                 # nf4 or fp4 depending on bitsandbytes version
        #     # bnb_4bit_compute_dtype=torch.bfloat16     # compute in bf16 for stability
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModel.from_pretrained(model_name, config=config, quantization_config=None,
                                          device_map='cuda:0', trust_remote_code=True, torch_dtype=torch.bfloat16,
                                          token='')

        model = PeftModel.from_pretrained(model, model_name)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, model_name + "-supervised")
        model = model.merge_and_unload()
        model.config.use_cache = False

        dtype = model.get_input_embeddings().weight.dtype

        self.config = model.config

        self.backbone = model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels, device=model.device, dtype=dtype)

        self.loss_fct = nn.CrossEntropyLoss()

        self.device = model.device

    def forward(self, input_ids, attention_mask, labels=None, **kwarg):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, -1])
        logits = self.classifier(pooled)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, path):
        self.backbone.save_pretrained(path)
        torch.save(self.classifier.state_dict(), os.path.join(path, "classifier.pt"))

tokenizer = AutoTokenizer.from_pretrained('McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp')

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)


dataset_path = "/mnt/hungpv/projects/MoEmb/dataset/data/banking77"
dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join(dataset_path, "train.csv"),
        "dev": os.path.join(dataset_path, "dev.csv")
    }
)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

model = LLM2VecForSequenceClassification(num_labels=77)

peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    # task_type=TaskType.CAUSAL_LM
)

# model = get_peft_model(model, peft_config)

model.backbone = get_peft_model(model.backbone, peft_config)

# model = PeftModel.from_pretrained(model, "/content/qlora_cls/best_model")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


training_args = TrainingArguments(
    output_dir="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/banking77",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    learning_rate=1e-5,
    weight_decay = 0.01,
    warmup_ratio=0.1,
    optim = "adamw_torch",
    lr_scheduler_type = "cosine",
    logging_steps=200,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    bf16=True,   # dùng bf16 nếu GPU hỗ trợ
    report_to="none"
)

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        self.model.save_pretrained(output_dir)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

tokenizer = AutoTokenizer.from_pretrained('McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp')

from datasets import load_dataset

# Use the 'dev' split as the test set, as requested
test_dataset = dataset["dev"]


test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to('cuda')
        attention_mask = batch["attention_mask"].to('cuda')
        labels = batch["labels"].to('cuda')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

f1_score_macro = f1_score(all_labels, all_preds, average="macro")
print(f"Test F1 Score (Macro): {f1_score_macro:.4f}")

precision = precision_score(all_labels, all_preds, average="macro")
print(f"Test Precision: {precision:.4f}")

recall = recall_score(all_labels, all_preds, average="macro")
print(f"Test Recall: {recall:.4f}")

model.save_pretrained("/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/banking77/best_model")

# import os
# import zipfile
# import shutil
# import json
#
# # Setup kaggle.json
# os.makedirs("/root/.kaggle", exist_ok=True)
# shutil.copy(kaggle_api_path + "/kaggle.json", "/root/.kaggle/kaggle.json")
# os.chmod("/root/.kaggle/kaggle.json", 0o600)
#
# dataset_name = "llm2vec-banking-checkpoint"  # tên sẽ xuất hiện trên Kaggle
# user = "winddao"           # thay bằng username của bạn
#
# # Tạo metadata
# metadata = {
#     "title": dataset_name,
#     "id": f"{user}/{dataset_name}",
#     "licenses": [{"name": "Apache-2.0"}]
# }
#
# with open("/qlora_cls/best_model/dataset-metadata.json", "w") as f:
#     json.dump(metadata, f)
#
# get_ipython().system('kaggle datasets create -p /qlora_cls/best_model -u')