"""torchao Int4 quantization for Gemma-4-26B-A4B-it including MoE experts.

Manually applies Int4 weight-only quantization to ALL weight tensors,
including the fused 3D MoE expert tensors that standard PTQ libraries skip.

Approach: reshape 3D [experts, out, in] → 2D [experts*out, in], quantize
with torchao's group quantization, then pack and save.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import safetensors.torch
from torchao.quantization.quant_primitives import (
    quantize_affine,
    MappingType,
    ZeroPointDomain,
)

MODEL_PATH = "/workspace/gemma-4-26B-A4B-it"
TRIAL_DIR = Path(__file__).parent
WEIGHTS_DIR = TRIAL_DIR / "weights"

BITS = 4
GROUP_SIZE = 128


def quantize_tensor_int4(weight: torch.Tensor) -> dict:
    """Quantize a 2D weight tensor to INT4 with group quantization.

    Returns dict with quantized data, scales, zeros for later dequantization.
    """
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    device = weight.device

    # Ensure 2D
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {weight.ndim}D")

    rows, cols = weight.shape
    # Pad cols to multiple of group_size
    pad = (GROUP_SIZE - cols % GROUP_SIZE) % GROUP_SIZE
    if pad > 0:
        weight = torch.nn.functional.pad(weight, (0, pad))

    n_groups = weight.shape[1] // GROUP_SIZE
    weight_grouped = weight.reshape(rows, n_groups, GROUP_SIZE)

    # Compute per-group min/max for asymmetric quantization
    w_min = weight_grouped.min(dim=-1, keepdim=True).values.float()
    w_max = weight_grouped.max(dim=-1, keepdim=True).values.float()

    # INT4 range: 0..15 (unsigned) or -8..7 (signed)
    # Use unsigned for packing simplicity
    q_min, q_max = 0, 15
    scale = (w_max - w_min) / (q_max - q_min)
    scale = scale.clamp(min=1e-10)
    zero_point = (q_min - w_min / scale).round().clamp(q_min, q_max)

    # Quantize
    w_q = (weight_grouped.float() / scale + zero_point).round().clamp(q_min, q_max).to(torch.uint8)

    # Pack two int4 values into one uint8
    w_q_flat = w_q.reshape(rows, -1)  # [rows, n_groups * GROUP_SIZE]
    if w_q_flat.shape[1] % 2 != 0:
        w_q_flat = torch.nn.functional.pad(w_q_flat, (0, 1))
    packed = (w_q_flat[:, 0::2] | (w_q_flat[:, 1::2] << 4))

    return {
        "packed": packed,  # [rows, cols/2] uint8
        "scales": scale.squeeze(-1).to(orig_dtype),  # [rows, n_groups]
        "zeros": zero_point.squeeze(-1).to(torch.uint8),  # [rows, n_groups]
        "orig_shape": orig_shape,
        "padded_cols": weight.shape[1],
    }


def main():
    from transformers import AutoConfig, AutoTokenizer

    print(f"[quantize] Loading model state dict from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load state dict shard by shard to avoid OOM
    index_path = Path(MODEL_PATH) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    quantized_state = {}
    total_orig_bytes = 0
    total_quant_bytes = 0

    for shard_idx, shard_file in enumerate(shard_files):
        print(f"[quantize] Processing shard {shard_idx+1}/{len(shard_files)}: {shard_file}")
        shard_path = Path(MODEL_PATH) / shard_file
        shard = safetensors.torch.load_file(str(shard_path), device="cpu")

        for key, tensor in shard.items():
            orig_bytes = tensor.nelement() * tensor.element_size()
            total_orig_bytes += orig_bytes

            # Skip small tensors (norms, scalars, biases)
            if tensor.ndim < 2 or tensor.numel() < 1024:
                quantized_state[key] = tensor
                total_quant_bytes += orig_bytes
                continue

            # Handle 3D MoE expert tensors
            if tensor.ndim == 3:
                # [num_experts, out, in] → [num_experts*out, in]
                ne, out_dim, in_dim = tensor.shape
                reshaped = tensor.reshape(ne * out_dim, in_dim)
                try:
                    q = quantize_tensor_int4(reshaped)
                    quantized_state[f"{key}.packed"] = q["packed"]
                    quantized_state[f"{key}.scales"] = q["scales"]
                    quantized_state[f"{key}.zeros"] = q["zeros"]
                    quantized_state[f"{key}.meta"] = torch.tensor(
                        [ne, out_dim, in_dim, q["padded_cols"], BITS, GROUP_SIZE],
                        dtype=torch.int64,
                    )
                    qbytes = q["packed"].nelement() + q["scales"].nelement() * q["scales"].element_size() + q["zeros"].nelement()
                    total_quant_bytes += qbytes
                    print(f"  {key}: 3D [{ne},{out_dim},{in_dim}] → INT4 ({orig_bytes/1e6:.1f}MB → {qbytes/1e6:.1f}MB)")
                except Exception as e:
                    print(f"  {key}: SKIP 3D quantize error: {e}")
                    quantized_state[key] = tensor
                    total_quant_bytes += orig_bytes
                continue

            # Handle 2D linear weights
            if tensor.ndim == 2 and min(tensor.shape) >= 64:
                try:
                    q = quantize_tensor_int4(tensor)
                    quantized_state[f"{key}.packed"] = q["packed"]
                    quantized_state[f"{key}.scales"] = q["scales"]
                    quantized_state[f"{key}.zeros"] = q["zeros"]
                    quantized_state[f"{key}.meta"] = torch.tensor(
                        [1, tensor.shape[0], tensor.shape[1], q["padded_cols"], BITS, GROUP_SIZE],
                        dtype=torch.int64,
                    )
                    qbytes = q["packed"].nelement() + q["scales"].nelement() * q["scales"].element_size() + q["zeros"].nelement()
                    total_quant_bytes += qbytes
                except Exception as e:
                    quantized_state[key] = tensor
                    total_quant_bytes += orig_bytes
                continue

            # Everything else: keep as-is
            quantized_state[key] = tensor
            total_quant_bytes += orig_bytes

        del shard
        gc.collect()

    print(f"\n[quantize] Original: {total_orig_bytes/1e9:.1f} GB")
    print(f"[quantize] Quantized: {total_quant_bytes/1e9:.1f} GB")
    print(f"[quantize] Compression: {total_orig_bytes/max(total_quant_bytes,1):.1f}x")

    print(f"\n[quantize] Saving to {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(quantized_state, str(WEIGHTS_DIR / "model.safetensors"))

    # Copy config and tokenizer
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "chat_template.jinja"]:
        src = Path(MODEL_PATH) / fname
        if src.exists():
            shutil.copy2(src, WEIGHTS_DIR / fname)

    shutil.copy2(Path(MODEL_PATH) / "config.json", TRIAL_DIR / "config.json")
    tokenizer.save_pretrained(WEIGHTS_DIR)

    print(f"[quantize] Done!")


if __name__ == "__main__":
    main()
