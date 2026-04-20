"""AIME 2026 evaluation — harness-owned protocol.

R5: sampling parameters (sample_n, max_tokens, ctx_size, temperature, top_p,
    top_k, prompt, chat_template, extractor, strict_parsing) are pinned in
    configs/pinned.yaml.
R15: quality gate (≥ 85.3%) is enforced by harness/verify/run.py.

Protocol (frozen):
    dataset        : MathArena/aime_2026 (I + II, 30 problems)
    n_problems     : 30
    sample_n       : 4             → total attempts = 120
    temperature    : 0.6
    top_p          : 0.95
    top_k          : unset
    max_tokens     : 64000
    ctx_size       : 72000
    chat_template  : Gemma 4 default render (thinking auto-included)
    strict_parsing : false
    extraction     : 1. balanced-brace \\boxed{N} / \\fbox{N} (last match)
                     2. fallback (strict_parsing=false): last \\b\\d+\\b in text
    scoring        : per-sample pass rate = correct / (30 × 4)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness.common import pinned, signing, trial  # noqa: E402

# ------------------------------------------------------------------------- #
# Extraction                                                                #
# ------------------------------------------------------------------------- #

_BOXED_MARKERS = ("\\boxed{", "\\fbox{")
_LAST_INT = re.compile(r"\b\d+\b")


def _iter_balanced_boxed(text: str) -> list[str]:
    """Yield contents of every balanced-brace \\boxed{...} / \\fbox{...}.

    Handles arbitrarily nested braces ("recursive match") — e.g.
    \\boxed{\\frac{1}{2}} returns "\\frac{1}{2}". For AIME we only care
    about the integer payload, but the matcher must be brace-balanced so a
    valid boxed answer inside a bigger LaTeX structure is still picked up.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        marker_len = 0
        for m in _BOXED_MARKERS:
            if text.startswith(m, i):
                marker_len = len(m)
                break
        if marker_len == 0:
            i += 1
            continue
        start = i + marker_len
        depth = 1
        j = start
        while j < n and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            out.append(text[start:j])
            i = j + 1
        else:
            # unbalanced — stop scanning from here to avoid O(n^2) pathologies
            break
    return out


def _to_int_in_range(s: str, lo: int, hi: int) -> int | None:
    # Strip LaTeX wrappers / whitespace, keep digits + optional leading sign.
    cleaned = re.sub(r"[^\d\-]", "", s)
    if not cleaned or cleaned in ("-", "--"):
        return None
    try:
        v = int(cleaned)
    except ValueError:
        return None
    return v if lo <= v <= hi else None


def extract_answer(text: str, strict_parsing: bool, answer_range=(0, 999)) -> int | None:
    lo, hi = answer_range
    boxed = _iter_balanced_boxed(text)
    for content in reversed(boxed):  # prefer the last boxed expression
        v = _to_int_in_range(content, lo, hi)
        if v is not None:
            return v
    if strict_parsing:
        return None
    # Fallback: last bare integer in the text.
    last = None
    for m in _LAST_INT.finditer(text):
        last = m.group(0)
    if last is None:
        return None
    try:
        v = int(last)
    except ValueError:
        return None
    return v if lo <= v <= hi else None


# ------------------------------------------------------------------------- #
# Dataset                                                                   #
# ------------------------------------------------------------------------- #

def _load_problems() -> list[dict]:
    """Load the frozen MathArena/aime_2026 snapshot.

    The snapshot is stored at harness/eval/aime_2026.json as
    [{id, problem, answer}, ...] with exactly 30 entries (AIME-I + AIME-II).
    Seed it via scripts/fetch_aime_2026.py — not from this module — so the
    eval runner has no network side effect at trial time.
    """
    path = REPO_ROOT / "harness" / "eval" / "aime_2026.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/fetch_aime_2026.py to freeze the "
            "MathArena/aime_2026 problem set before evaluation."
        )
    data = json.loads(path.read_text())
    cfg = pinned.load()["eval"]
    if len(data) != cfg["n_problems"]:
        raise ValueError(
            f"aime_2026.json has {len(data)} problems, expected {cfg['n_problems']}."
        )
    for row in data:
        if set(row) < {"id", "problem", "answer"}:
            raise ValueError(f"aime_2026.json row missing fields: {row}")
    return data


# ------------------------------------------------------------------------- #
# Runner                                                                    #
# ------------------------------------------------------------------------- #

def _render_prompt(tokenizer, problem_text: str, template: str) -> str:
    """Apply the tokenizer chat template (Gemma 4 default render).

    The user turn is the pinned prompt with the problem substituted. Gemma 4
    renders thinking automatically when `add_generation_prompt=True`.
    """
    user_msg = template.format(problem=problem_text)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


def run(trial_id: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = pinned.load()["eval"]
    tdir = trial.trial_dir(trial_id)
    intent = trial.load_intent(trial_id)
    weights_path = tdir / "weights"

    problems = _load_problems()

    tokenizer = AutoTokenizer.from_pretrained(weights_path)
    model = AutoModelForCausalLM.from_pretrained(
        weights_path,
        torch_dtype=getattr(torch, cfg["dtype"]),
        attn_implementation=cfg["attn_implementation"],
        device_map={"": 0},
    )
    model.eval()

    torch.manual_seed(cfg["seed"])

    gen_kwargs = dict(
        max_new_tokens=cfg["max_tokens"],
        do_sample=True,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
    )
    if cfg.get("top_k") is not None:
        gen_kwargs["top_k"] = cfg["top_k"]

    per_problem: list[dict] = []
    total_attempts = cfg["n_problems"] * cfg["sample_n"]
    total_correct = 0

    for p in problems:
        prompt = _render_prompt(tokenizer, p["problem"], cfg["prompt_template"])
        ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg["ctx_size"] - cfg["max_tokens"],
        ).input_ids.to("cuda:0")

        samples: list[dict] = []
        problem_correct = 0
        for _ in range(cfg["sample_n"]):
            with torch.inference_mode():
                out = model.generate(ids, **gen_kwargs)
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            pred = extract_answer(text, strict_parsing=cfg["strict_parsing"])
            correct = pred is not None and int(pred) == int(p["answer"])
            if correct:
                problem_correct += 1
                total_correct += 1
            samples.append({"pred": pred, "correct": correct})
        per_problem.append(
            {
                "id": p["id"],
                "gold": p["answer"],
                "correct_samples": problem_correct,
                "samples": samples,
            }
        )

    accuracy = total_correct / total_attempts
    gate = pinned.load()["quality_gate"]["aime_2026_no_tools_min"]

    payload = {
        "trial_id": trial_id,
        "engine": intent.get("engine", "transformers"),
        "benchmark": "AIME_2026_no_tools",
        "protocol": cfg["protocol"],
        "dataset": cfg["dataset"],
        "pinned": cfg,
        "n_problems": cfg["n_problems"],
        "sample_n": cfg["sample_n"],
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "accuracy": accuracy,                  # fraction in [0, 1]
        "accuracy_percent": accuracy * 100.0,  # convenience
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
