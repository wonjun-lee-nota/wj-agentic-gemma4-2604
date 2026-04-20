"""AIME 2026 evaluation — harness-owned protocol.

R5: sampling parameters (sample_n, max_tokens, temperature, extractor) are pinned.
R15: quality gate (≥ 85.3%) is enforced here; trial fails if below threshold.

Defaults follow MathArena conventions: boxed-answer extractor, low-temperature
majority over N samples. The agent cannot swap the extractor at runtime.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness.common import pinned, signing, trial  # noqa: E402

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_boxed(text: str) -> str | None:
    matches = BOXED.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip()
    # AIME answers are integers 000-999. Normalize.
    digits = re.sub(r"[^0-9]", "", ans)
    return digits[-3:] if digits else None


def _load_problems(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def run(trial_id: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = pinned.load()["eval"]
    tdir = trial.trial_dir(trial_id)
    intent = trial.load_intent(trial_id)
    weights_path = tdir / "weights"

    problems_path = REPO_ROOT / "harness" / "eval" / "aime_2026.json"
    if not problems_path.exists():
        raise FileNotFoundError(
            "harness/eval/aime_2026.json missing — seed the problem set before running."
        )
    problems = _load_problems(problems_path)

    tokenizer = AutoTokenizer.from_pretrained(weights_path)
    model = AutoModelForCausalLM.from_pretrained(
        weights_path,
        torch_dtype=getattr(torch, cfg["dtype"]),
        attn_implementation=cfg["attn_implementation"],
        device_map={"": 0},
    )
    model.eval()

    per_problem: list[dict] = []
    correct = 0
    torch.manual_seed(cfg["seed"])

    for p in problems:
        samples: list[str | None] = []
        for i in range(cfg["sample_n"]):
            prompt = cfg["prompt_template"].format(problem=p["problem"])
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
            with torch.inference_mode():
                out = model.generate(
                    ids,
                    max_new_tokens=cfg["max_tokens"],
                    do_sample=cfg["temperature"] > 0,
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                )
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            samples.append(extract_boxed(text))
        voted = Counter([s for s in samples if s is not None]).most_common(1)
        pred = voted[0][0] if voted else None
        is_correct = pred is not None and pred == p["answer"]
        correct += int(is_correct)
        per_problem.append(
            {"id": p["id"], "pred": pred, "gold": p["answer"], "correct": is_correct}
        )

    accuracy = correct / len(problems)
    gate = pinned.load()["quality_gate"]["aime_2026_no_tools_min"]
    payload = {
        "trial_id": trial_id,
        "engine": intent.get("engine", "transformers"),
        "benchmark": "AIME_2026_no_tools",
        "protocol": "MathArena-style boxed extractor, majority over sample_n",
        "pinned": cfg,
        "num_problems": len(problems),
        "accuracy": accuracy,
        "pass_gate": accuracy >= gate,
        "gate_threshold": gate,
        "per_problem": per_problem,
    }
    signed = signing.sign_payload(trial_id, intent.get("engine", "transformers"), payload)
    (tdir / "eval.json").write_text(json.dumps(signed, indent=2))
    return signed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    args = ap.parse_args()
    run(args.trial)
    print(f"[eval] wrote {trial.trial_dir(args.trial) / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
