from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import shutil
import zipfile
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

def is_probably_zip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig == b"PK\x03\x04"
    except Exception:
        return False

def try_extract_zip(zf_path: Path, extract_root: Path) -> bool:
    # Thử với zipfile
    try:
        with zipfile.ZipFile(zf_path, "r") as z:
            z.extractall(extract_root)
        return True
    except Exception as e1:
        # Thử shutil (unpack_archive)
        try:
            shutil.unpack_archive(str(zf_path), str(extract_root))
            return True
        except Exception as e2:
            # Thử system unzip nếu có
            if shutil.which("unzip"):
                try:
                    subprocess.run(
                        ["unzip", "-o", str(zf_path), "-d", str(extract_root)],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    return True
                except Exception as e3:
                    print(f"[!] unzip system failed: {e3}")
            print(f"[!] Cannot extract '{zf_path.name}': zipfile={e1} | shutil={e2}")
            return False

def download_and_unzip_repo(
    repo_id: str = "HoangTran223/bert_dskd_baseline",
    target_dir: str = "/mnt/hungpv/projects/MoEmb/model_hub/bert_dskd_baseline",
    cache_dir: str = None,
    allow_patterns=None,
    ignore_patterns=None,
    force_redownload: bool = False
):
    os.makedirs(target_dir, exist_ok=True)
    print(f"[+] Downloading snapshot of {repo_id} ...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_files_only=False,
        force_download=force_redownload,
        repo_type="model",
        local_dir_use_symlinks=False,  # đảm bảo không phải symlink tới file khác
    )
    print(f"[+] Snapshot cached at: {snapshot_path}")

    snapshot_path = Path(snapshot_path)
    zip_files, other_files = [], []
    for p in snapshot_path.rglob("*"):
        if p.is_file():
            if p.suffix.lower() == ".zip":
                zip_files.append(p)
            else:
                other_files.append(p)

    print(f"[+] Found {len(zip_files)} zip files, {len(other_files)} other files.")

    # Giải nén
    for zf in zip_files:
        size = zf.stat().st_size
        rel_parent = zf.parent.relative_to(snapshot_path)
        extract_root = Path(target_dir) / rel_parent / zf.stem
        if size < 512:
            print(f"[!] Skip '{zf.name}' (too small {size} bytes, có thể là pointer LFS).")
            continue
        os.makedirs(extract_root, exist_ok=True)
        if not is_probably_zip(zf):
            print(f"[!] '{zf.name}' không có magic ZIP (PK\\x03\\x04). Giữ nguyên file gốc.")
            # copy thẳng file về (không giải nén)
            dest_path = Path(target_dir) / rel_parent / zf.name
            if not dest_path.exists():
                shutil.copy2(zf, dest_path)
            continue
        ok = try_extract_zip(zf, extract_root)
        if ok:
            print(f"[✓] Unzipped: {zf.relative_to(snapshot_path)} -> {extract_root}")
        else:
            print(f"[!] Failed to extract '{zf.name}'. Copy raw file.")
            dest_path = Path(target_dir) / rel_parent / zf.name
            if not dest_path.exists():
                shutil.copy2(zf, dest_path)

    # Copy file khác
    for f in other_files:
        rel_path = f.relative_to(snapshot_path)
        dest_path = Path(target_dir) / rel_path
        os.makedirs(dest_path.parent, exist_ok=True)
        if not dest_path.exists():
            shutil.copy2(f, dest_path)

    print(f"[+] Done. All content stored under: {target_dir}")

if __name__ == "__main__":
    download_and_unzip_repo(
        repo_id="HoangTran223/patent_teacher",
        target_dir="/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/patent"
    )


