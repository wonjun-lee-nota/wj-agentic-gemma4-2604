"""GPTQ W4 G128 quantization with unfused MoE experts for Gemma-4-26B-A4B-it.

Standard quantization libraries cannot handle Gemma 4's fused 3D MoE expert
tensors [num_experts, out, in]. This script:
1. Loads the BF16 model
2. Unfuses experts into individual Linear layers
3. Runs GPTQ on the unfused model (all linear layers including experts)
4. Saves the quantized model

This is the only way to get true 4-bit quantization on this MoE architecture
until quantization libraries add native fused-MoE support.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import shutil
from pathlib import Path

import torch
from gptqmodel import GPTQModel, QuantizeConfig

MODEL_PATH = "/workspace/gemma-4-26B-A4B-it"
TRIAL_DIR = Path(__file__).parent
WEIGHTS_DIR = TRIAL_DIR / "weights"

BITS = 4
GROUP_SIZE = 128
CALIBRATION_SAMPLES = 256
CALIBRATION_SEQLEN = 2048


def main():
    print("[quantize] Setting up GPTQ config: bits=4, group_size=128")
    quant_config = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        damp_percent=0.01,
    )

    print(f"[quantize] Loading model from {MODEL_PATH}")
    model = GPTQModel.load(MODEL_PATH, quant_config)

    # Unfuse MoE experts: convert 3D fused tensors to individual Linear layers
    print("[quantize] Unfusing MoE experts...")
    unfused_count = 0
    lang_model = model.model.language_model if hasattr(model.model, 'language_model') else model.model
    for layer_idx, layer in enumerate(lang_model.layers):
        if not hasattr(layer, 'experts'):
            continue

        experts_module = layer.experts
        # Check for fused 3D gate_up_proj and down_proj
        if hasattr(experts_module, 'gate_up_proj') and isinstance(experts_module.gate_up_proj, torch.nn.Parameter):
            gate_up = experts_module.gate_up_proj.data  # [num_experts, 2*moe_intermediate, hidden]
            down = experts_module.down_proj.data  # [num_experts, hidden, moe_intermediate]
            num_experts = gate_up.shape[0]

            # Create individual expert linear layers
            for e in range(num_experts):
                # gate_up_proj: split into gate and up
                gu_weight = gate_up[e]  # [2*moe_intermediate, hidden]
                mid = gu_weight.shape[0] // 2
                gate_w = gu_weight[:mid]  # [moe_intermediate, hidden]
                up_w = gu_weight[mid:]    # [moe_intermediate, hidden]
                down_w = down[e]          # [hidden, moe_intermediate]

                # Register as named Linear modules so GPTQ can find them
                gate_lin = torch.nn.Linear(gate_w.shape[1], gate_w.shape[0], bias=False, device='meta')
                gate_lin.weight = torch.nn.Parameter(gate_w)
                up_lin = torch.nn.Linear(up_w.shape[1], up_w.shape[0], bias=False, device='meta')
                up_lin.weight = torch.nn.Parameter(up_w)
                down_lin = torch.nn.Linear(down_w.shape[1], down_w.shape[0], bias=False, device='meta')
                down_lin.weight = torch.nn.Parameter(down_w)

                setattr(experts_module, f'expert_{e}_gate', gate_lin)
                setattr(experts_module, f'expert_{e}_up', up_lin)
                setattr(experts_module, f'expert_{e}_down', down_lin)

            # Remove original fused tensors
            del experts_module.gate_up_proj
            del experts_module.down_proj
            unfused_count += num_experts
            print(f"  Layer {layer_idx}: unfused {num_experts} experts")

    print(f"[quantize] Unfused {unfused_count} total experts")
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare calibration data
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

    # Copy config.json
    config_src = WEIGHTS_DIR / "config.json"
    config_dst = TRIAL_DIR / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, config_dst)
    print(f"[quantize] Done. Weights saved to {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
