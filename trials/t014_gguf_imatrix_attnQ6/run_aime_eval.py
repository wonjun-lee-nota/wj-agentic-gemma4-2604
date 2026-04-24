"""AIME 2026 evaluation via llama.cpp server — follows pinned.yaml protocol exactly.

Calls the llama.cpp OpenAI-compatible /v1/chat/completions endpoint.
Implements the same extraction/scoring as harness/eval/aime_runner.py.
Results are signed via the harness signing module for verify compatibility.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness.common import pinned, signing, trial  # noqa: E402

SERVER_URL = "http://127.0.0.1:8080"
TRIAL_ID = "t014_gguf_imatrix_attnQ6"

# ── Extraction (identical to harness/eval/aime_runner.py) ────────────────

_BOXED_MARKERS = ("\\boxed{", "\\fbox{")
_LAST_INT = re.compile(r"\b\d+\b")


def _iter_balanced_boxed(text: str) -> list[str]:
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
            break
    return out


def _to_int_in_range(s: str, lo: int, hi: int) -> int | None:
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
    for content in reversed(boxed):
        v = _to_int_in_range(content, lo, hi)
        if v is not None:
            return v
    if strict_parsing:
        return None
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


# ── Dataset ──────────────────────────────────────────────────────────────

def load_problems() -> list[dict]:
    path = REPO_ROOT / "harness" / "eval" / "aime_2026.json"
    data = json.loads(path.read_text())
    return data


# ── API call ─────────────────────────────────────────────────────────────

def generate_response(problem_text: str, cfg: dict) -> str:
    """Call llama.cpp server with the pinned AIME protocol parameters."""
    prompt_template = cfg["prompt_template"]
    # Use replace instead of .format() because problem text contains LaTeX braces
    user_msg = prompt_template.replace("{problem}", problem_text)

    payload = {
        "model": "gemma4",
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [retry {attempt+1}/3] API error: {e}")
            time.sleep(5)
    return ""


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    cfg = pinned.load()["eval"]
    problems = load_problems()
    intent = trial.load_intent(TRIAL_ID)
    tdir = trial.trial_dir(TRIAL_ID)

    print(f"[aime] Starting AIME 2026 eval: {cfg['n_problems']} problems × {cfg['sample_n']} samples")
    print(f"[aime] Protocol: T={cfg['temperature']}, top_p={cfg['top_p']}, max_tokens={cfg['max_tokens']}")

    per_problem: list[dict] = []
    total_attempts = cfg["n_problems"] * cfg["sample_n"]
    total_correct = 0

    for pi, p in enumerate(problems):
        print(f"\n[aime] Problem {pi+1}/{len(problems)}: {p['id']} (answer={p['answer']})")
        samples: list[dict] = []
        problem_correct = 0

        for si in range(cfg["sample_n"]):
            print(f"  Sample {si+1}/{cfg['sample_n']}...", end=" ", flush=True)
            t0 = time.time()
            text = generate_response(p["problem"], cfg)
            elapsed = time.time() - t0

            pred = extract_answer(text, strict_parsing=cfg["strict_parsing"])
            correct = pred is not None and int(pred) == int(p["answer"])
            if correct:
                problem_correct += 1
                total_correct += 1

            status = "✓" if correct else "✗"
            print(f"pred={pred} {status} ({elapsed:.1f}s)")
            samples.append({"pred": pred, "correct": correct})

        per_problem.append({
            "id": p["id"],
            "gold": p["answer"],
            "correct_samples": problem_correct,
            "samples": samples,
        })
        running_acc = total_correct / ((pi + 1) * cfg["sample_n"])
        print(f"  Problem score: {problem_correct}/{cfg['sample_n']} | Running accuracy: {running_acc:.1%}")

    accuracy = total_correct / total_attempts
    gate = pinned.load()["quality_gate"]["aime_2026_no_tools_min"]

    print(f"\n{'='*60}")
    print(f"[aime] FINAL: {total_correct}/{total_attempts} = {accuracy:.1%}")
    print(f"[aime] Gate: {'PASS' if accuracy >= gate else 'FAIL'} (threshold: {gate:.1%})")
    print(f"{'='*60}")

    payload = {
        "trial_id": TRIAL_ID,
        "engine": intent.get("engine", "vllm"),
        "benchmark": "AIME_2026_no_tools",
        "protocol": cfg["protocol"],
        "dataset": cfg["dataset"],
        "pinned": cfg,
        "n_problems": cfg["n_problems"],
        "sample_n": cfg["sample_n"],
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "pass_gate": accuracy >= gate,
        "gate_threshold": gate,
        "per_problem": per_problem,
    }
    signed = signing.sign_payload(TRIAL_ID, intent.get("engine", "vllm"), payload)
    out_path = tdir / "eval.json"
    out_path.write_text(json.dumps(signed, indent=2))
    print(f"\n[aime] Wrote {out_path}")


if __name__ == "__main__":
    main()
