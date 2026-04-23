"""Prepare calibration text for llama.cpp imatrix generation.

Exports C4 text samples to a plain text file that llama-imatrix can consume.
"""
from datasets import load_dataset

OUT = "/tmp/imatrix_calibration.txt"
N_SAMPLES = 256
MAX_CHARS = 4096

print(f"[prep] Loading C4 calibration data ({N_SAMPLES} samples)...")
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

texts = []
for sample in ds:
    text = sample["text"].strip()
    if len(text) >= 512:
        texts.append(text[:MAX_CHARS])
    if len(texts) >= N_SAMPLES:
        break

with open(OUT, "w") as f:
    for t in texts:
        f.write(t + "\n")

print(f"[prep] Wrote {len(texts)} samples to {OUT}")
