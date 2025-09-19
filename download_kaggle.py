import kagglehub

# Download latest version
path = kagglehub.dataset_download("winddao/llm2vec-sts-12-checkpoint")



print("Path to dataset files:", path)
import shutil
import os

# Đường dẫn đích
target_dir = "/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/sts12"

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(target_dir, exist_ok=True)

# Di chuyển tất cả file/folder từ path vào target_dir
for item in os.listdir(path):
	src = os.path.join(path, item)
	dst = os.path.join(target_dir, item)
	if os.path.isdir(src):
		shutil.copytree(src, dst, dirs_exist_ok=True)
	else:
		shutil.copy2(src, dst)