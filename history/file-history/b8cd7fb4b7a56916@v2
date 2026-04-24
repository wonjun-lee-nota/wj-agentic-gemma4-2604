"""GPTQ W4 G128 quantization for Gemma-4-26B-A4B-it.

Executor-owned script — not part of the sealed harness.
Uses gptqmodel directly (not transformers GPTQConfig) for Gemma 4 compatibility.
"""
from __future__ import annotations

import shutil
from pathlib import Path

MODEL_PATH = "/workspace/gemma-4-26B-A4B-it"
TRIAL_DIR = Path(__file__).parent
WEIGHTS_DIR = TRIAL_DIR / "weights"

BITS = 4
GROUP_SIZE = 128
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQLEN = 2048


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from gptqmodel import GPTQModel, QuantizeConfig

    print(f"[quantize] Setting up GPTQ config: bits={BITS}, group_size={GROUP_SIZE}")
    quant_config = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        damp_percent=0.01,
    )

    print(f"[quantize] Loading model from {MODEL_PATH}")
    model = GPTQModel.load(
        MODEL_PATH,
        quant_config,
    )

    print(f"[quantize] Preparing C4 calibration data ({CALIBRATION_SAMPLES} samples)")
    from datasets import load_dataset
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    calibration_data = []
    for sample in dataset:
        text = sample["text"]
        tokens = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=CALIBRATION_SEQLEN)
        if tokens.input_ids.shape[1] >= CALIBRATION_SEQLEN // 2:
            calibration_data.append({"input_ids": tokens.input_ids[0], "attention_mask": tokens.attention_mask[0]})
        if len(calibration_data) >= CALIBRATION_SAMPLES:
            break
    print(f"[quantize] Got {len(calibration_data)} calibration samples")

    print("[quantize] Starting quantization...")
    model.quantize(calibration_data)

    print(f"[quantize] Saving quantized model to {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(WEIGHTS_DIR)

    # Copy config.json to trial root for verification
    config_src = WEIGHTS_DIR / "config.json"
    config_dst = TRIAL_DIR / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, config_dst)
    print(f"[quantize] Done. Weights saved to {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
