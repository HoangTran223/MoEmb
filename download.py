# huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir /mnt/nam_x/tue_x/model_hub/mistral --resume-download --repo-type model --cache-dir /mnt/nam_x/tue_x/model_hub/mistral


from transformers import AutoTokenizer, AutoModelForCausalLM

def load_and_save_opt13b(target_dir: str, model_name: str):
    # Load từ HF Hub
    print(f"Loading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        #cache_dir=cache_dir  # Chỉ định thư mục cache
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        #cache_dir=cache_dir  # Chỉ định thư mục cache
    )

    # Save model về local
    print(f"Saving model and tokenizer to {target_dir} ...")
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    print("Done!")


load_and_save_opt13b(
    target_dir= "/mnt/hungpv/projects/MoEmb/lora_path/llm2vec",
    model_name= "HoangTran223/LLM2Vec-Mistral-7B-Instruct-v2-mntp_Banking77",
)


