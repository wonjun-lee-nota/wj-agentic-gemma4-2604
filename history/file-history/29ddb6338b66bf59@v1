"""Freeze MathArena/aime_2026 (I + II) into harness/eval/aime_2026.json.

Run once per repo. After this runs, the eval runner has no network
dependency at trial time — the frozen JSON is the authoritative snapshot
and the verifier compares every trial against the same 30 problems.

The repo author runs this; the executor agent should not. If the dataset
changes upstream, re-freezing is a deliberate, logged harness edit
(ALLOW_HARNESS_EDIT=<reason>) because it invalidates cross-session
comparability.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "harness" / "eval" / "aime_2026.json"
DATASET_ID = "MathArena/aime_2026"
EXPECTED_N = 30


def main() -> int:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(
            "datasets not installed. `pip install datasets` then re-run.",
            file=sys.stderr,
        )
        return 2

    ds = load_dataset(DATASET_ID)
    # MathArena publishes AIME-I and AIME-II as separate splits or as a
    # single merged split depending on version. Accept both shapes.
    rows: list[dict] = []
    if hasattr(ds, "keys") and set(ds.keys()) >= {"aime_i", "aime_ii"}:
        for split in ("aime_i", "aime_ii"):
            for r in ds[split]:
                rows.append(r)
    else:
        # Single split fallback.
        split_name = next(iter(ds.keys())) if hasattr(ds, "keys") else None
        source = ds[split_name] if split_name else ds
        for r in source:
            rows.append(r)

    # Normalize to {id, problem, answer}. The fields MathArena exposes may be
    # named differently across releases; map the common variants.
    normalized: list[dict] = []
    for r in rows:
        pid = r.get("id") or r.get("problem_id") or r.get("name")
        prob = r.get("problem") or r.get("question") or r.get("prompt")
        ans = r.get("answer") or r.get("gold") or r.get("solution")
        if pid is None or prob is None or ans is None:
            raise ValueError(f"row missing required field: {r}")
        try:
            ans_int = int(str(ans).strip())
        except ValueError:
            raise ValueError(f"answer not an int: {ans!r} in {pid!r}")
        if not (0 <= ans_int <= 999):
            raise ValueError(f"answer out of AIME range: {ans_int} in {pid!r}")
        normalized.append({"id": str(pid), "problem": prob, "answer": ans_int})

    if len(normalized) != EXPECTED_N:
        raise ValueError(
            f"expected {EXPECTED_N} problems (I + II), got {len(normalized)}"
        )

    OUT.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))
    print(f"wrote {OUT} with {len(normalized)} problems from {DATASET_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
