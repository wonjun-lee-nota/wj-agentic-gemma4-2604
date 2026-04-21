"""BitsAndBytes NF4 quantization for Gemma-4-26B-A4B-it.

Executor-owned script — not part of the sealed harness.
NF4 is a zero-shot method: no calibration data needed.
Quantization happens at load time via BitsAndBytesConfig.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/workspace/gemma-4-26B-A4B-it"
TRIAL_DIR = Path(__file__).parent
WEIGHTS_DIR = TRIAL_DIR / "weights"


def main():
    print("[quantize] Setting up BitsAndBytes NF4 config")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[quantize] Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"[quantize] Loading model with NF4 quantization from {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # spread across GPUs during NF4 load
    )

    print(f"[quantize] Saving quantized model to {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(WEIGHTS_DIR)
    tokenizer.save_pretrained(WEIGHTS_DIR)

    # Copy config.json to trial root for verification
    config_src = WEIGHTS_DIR / "config.json"
    config_dst = TRIAL_DIR / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, config_dst)
    print(f"[quantize] Done. Weights saved to {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
