# LLM2Vec_Distillation
Distillation for 3 tasks Classification, SentencePair and STS

1. Code preparation
Environent: python 3.11, multi-gpu or single gpu
pip install -r requirements.txt

2. download data.zip and place it under LLM2Vec_Distillation

3. prepare student and teacher
Download bert & LLM2vec, put them under folder model_hub inside LLM2Vec_Distillation

folder tree looks like:

LLM2Vec_Distillation
    |Classification
    |README.md
    |STS
    |SentencePair
    |configs
    |data
        |SICK
        |STS12
        |anli_r2
        |banking77
        |control
        |imdb
        |patent
        |scitail
        |stsbenchmark
    |requirements.txt
    |scripts
    |model_hub
        |bert
            |config.json
            |model.safetensors
            |special_tokens_map.json
            |tokenizer_config.json
            |tokenizer.json
            |vocab.txt
        |LLM2Vec
            |adapter_config.json
            |adapter_model.safetensors
            |config.json
            |special_tokens_map.json
            |tokenizer_config.json
            |tokenizer.json
            |tokenizer.model
    |outputs

3. For each task
    3.1 First supervise fine tune teacher, then the output teacher checkpoint will be exported into folder outputs
        cd /path to LLM2Vec_Distillation
        bash scripts/Classification/sft/banking77/banking77_LLM2Vec.sh
    3.2 KD bert student with sft teacher
        bash scripts/Classification/kd/banking77/banking77_ot_pro_rmse_cka.sh