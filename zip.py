import os
import zipfile
import shutil
import subprocess
import tarfile
from pathlib import Path

SRC_ZIP = Path("/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/patent/patent.zip")
DEST_DIR = Path("/mnt/hungpv/projects/MoEmb/lora_path/llm2vec/patent")

def debug_head(path: Path, n=512):
    try:
        with open(path, "rb") as f:
            head = f.read(n)
        print(f"[DEBUG] First {n} bytes (showing up to 64): {head[:64]}")
        # Kiểm tra magic TAR tại offset 257
        if len(head) >= 265 and head[257:262] in (b"ustar", b"ustar\x00"):
            print("[DEBUG] Detected possible TAR archive (ustar magic).")
    except Exception as e:
        print(f"[DEBUG] Cannot read head: {e}")

def try_zipfile(src: Path, dest: Path):
    try:
        with zipfile.ZipFile(src, "r") as z:
            bad = z.testzip()
            if bad:
                print(f"[WARN] Corrupted ZIP member: {bad}")
            z.extractall(dest)
        print(f"[✓] zipfile extracted: {src.name}")
        return True
    except Exception as e:
        print(f"[INFO] zipfile failed ({e})")
        return False

def try_shutil(src: Path, dest: Path):
    try:
        shutil.unpack_archive(str(src), str(dest))
        print(f"[✓] shutil.unpack_archive extracted: {src.name}")
        return True
    except Exception as e:
        print(f"[INFO] shutil.unpack_archive failed ({e})")
        return False

def try_system_unzip(src: Path, dest: Path):
    if not shutil.which("unzip"):
        print("[INFO] unzip command not found")
        return False
    try:
        r = subprocess.run(
            ["unzip", "-o", str(src), "-d", str(dest)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if r.returncode == 0:
            print(f"[✓] system unzip extracted: {src.name}")
            return True
        else:
            print(f"[INFO] system unzip failed (code={r.returncode}): {r.stderr[:200]}")
            return False
    except Exception as e:
        print(f"[INFO] system unzip exception: {e}")
        return False

def try_tar(src: Path, dest: Path):
    # Thử mở như TAR (nhiều file .tar bị đặt sai đuôi .zip)
    try:
        with tarfile.open(src, "r:*") as t:
            t.extractall(dest)
        print(f"[✓] tarfile extracted (misnamed?): {src.name}")
        return True
    except Exception as e:
        print(f"[INFO] tarfile failed ({e})")
        return False

def extract(src: Path, dest: Path):
    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {src}")
    size = src.stat().st_size
    print(f"[INFO] File: {src} | Size: {size/1024/1024:.2f} MB")
    debug_head(src)
    dest.mkdir(parents=True, exist_ok=True)

    # Thử lần lượt các phương pháp
    if try_zipfile(src, dest): return
    if try_shutil(src, dest): return
    if try_system_unzip(src, dest): return
    if try_tar(src, dest): return

    raise RuntimeError(f"Giải nén thất bại mọi phương pháp: {src}")

if __name__ == "__main__":
    extract(SRC_ZIP, DEST_DIR)
