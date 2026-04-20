"""Performance benchmark — the ONLY sanctioned path for TTFT / tok-per-sec /
p50·p95 / weight_size numbers.

R5: measurement logic is harness-owned. Parameters (input_len=128, output_len=128,
batch=1, warmup=5, runs=20) are pinned and cannot be overridden by flags.
R7: model.generate() for latency is forbidden outside this module; the policy
hook detects ad-hoc scripts that call .generate() and blocks them.

Inference is pinned to a single RTX 3090 (CUDA device 0). Multi-GPU during
quantization is permitted separately, but measurement must be single-card so
that the TTFT / tok-per-sec numbers reflect the target deployment constraint.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness.common import pinned, signing, trial  # noqa: E402

SINGLE_CARD_DEVICE = "0"


def _pin_single_card() -> None:
    # Hook-enforced too, but belt-and-braces: measurement must not see > 1 GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = SINGLE_CARD_DEVICE


def _weight_size_bytes(weights_dir: Path) -> int:
    total = 0
    for p in weights_dir.rglob("*"):
        if p.is_file() and p.suffix in {".safetensors", ".bin", ".pt", ".gguf"}:
            total += p.stat().st_size
    return total


def _run_one(model, tokenizer, input_ids, max_new_tokens: int) -> tuple[float, float]:
    """Return (ttft_ms, decode_tokens_per_sec) for one run.

    TTFT is measured as the wall time until the first generated token callback.
    Tokens/sec is computed over the remaining decoded tokens.
    """
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ttft = {"value": None}
    tokens_after_first: list[float] = []

    def on_token(_generated_ids):
        if ttft["value"] is None:
            torch.cuda.synchronize()
            ttft["value"] = (time.perf_counter() - t0) * 1000.0
        tokens_after_first.append(time.perf_counter())

    # Minimal custom decode loop — avoids HF generate() heuristics, gives a
    # deterministic, comparable number across quant engines.
    past = None
    cur_ids = input_ids
    generated = 0
    while generated < max_new_tokens:
        out = model(cur_ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        on_token(next_tok)
        cur_ids = next_tok
        generated += 1
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_ms = (t_end - t0) * 1000.0
    decode_ms = total_ms - ttft["value"]
    decode_tokps = (max_new_tokens - 1) / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    return ttft["value"], decode_tokps


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run(trial_id: str) -> dict:
    _pin_single_card()
    cfg = pinned.load()["metrics"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tdir = trial.trial_dir(trial_id)
    intent = trial.load_intent(trial_id)
    weights_path = tdir / "weights"

    tokenizer = AutoTokenizer.from_pretrained(weights_path)
    model = AutoModelForCausalLM.from_pretrained(
        weights_path,
        torch_dtype=getattr(torch, cfg["dtype"]),
        attn_implementation=cfg["attn_implementation"],
        device_map={"": 0},
    )
    model.eval()

    prompt = "A" * cfg["input_tokens"]  # placeholder; pinned prompt set lives beside this
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    input_ids = input_ids[:, : cfg["input_tokens"]]

    ttfts, tokps, totals = [], [], []
    with torch.inference_mode():
        for _ in range(cfg["warmup"]):
            _run_one(model, tokenizer, input_ids, cfg["output_tokens"])
        for _ in range(cfg["runs"]):
            t0 = time.perf_counter()
            ttft, tps = _run_one(model, tokenizer, input_ids, cfg["output_tokens"])
            totals.append((time.perf_counter() - t0) * 1000.0)
            ttfts.append(ttft)
            tokps.append(tps)

    payload = {
        "trial_id": trial_id,
        "engine": intent.get("engine", "transformers"),
        "method": intent["method"],
        "bit_width": intent["bit_width"],
        "pinned": cfg,
        "weight_size_bytes": _weight_size_bytes(weights_path),
        "ttft_ms": {
            "mean": statistics.mean(ttfts),
            "p50": _percentile(ttfts, 0.50),
            "p95": _percentile(ttfts, 0.95),
        },
        "decode_tokens_per_sec": {
            "mean": statistics.mean(tokps),
            "p50": _percentile(tokps, 0.50),
            "p95": _percentile(tokps, 0.95),
        },
        "total_latency_ms": {
            "p50": _percentile(totals, 0.50),
            "p95": _percentile(totals, 0.95),
        },
        "samples": {"ttft_ms": ttfts, "tok_per_sec": tokps, "total_ms": totals},
    }
    signed = signing.sign_payload(trial_id, intent.get("engine", "transformers"), payload)
    (tdir / "metrics.json").write_text(json.dumps(signed, indent=2))
    return signed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    args = ap.parse_args()
    run(args.trial)
    print(f"[metrics] wrote {trial.trial_dir(args.trial) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
