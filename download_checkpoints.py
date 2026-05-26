"""
download_checkpoints.py — Downloads MINT model checkpoints from HuggingFace.
Usage: python download_checkpoints.py [target_dir]
"""
import sys
import requests
from pathlib import Path
from tqdm import tqdm

BASE = "https://huggingface.co/varunullanat2012/mint/resolve/main"
FILES = [
    ("mint.ckpt",       "MINT backbone checkpoint (3.25 GB)"),
    ("bernett_mlp.pth", "Bernett MLP classifier   (26 MB)"),
]

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent


def download(filename: str, desc: str) -> None:
    dest = TARGET / filename
    if dest.exists():
        print(f"  {filename} — already exists, skipping.")
        return
    print(f"  {desc}")
    r = requests.get(f"{BASE}/{filename}", stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with dest.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                      ncols=70, desc=filename) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved → {dest}")


if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    for fname, fdesc in FILES:
        download(fname, fdesc)
    print("\nAll checkpoints ready.")
