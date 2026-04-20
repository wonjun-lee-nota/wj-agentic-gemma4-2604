"""Aggregate the session report (R8, R9, R11, R15).

Refuses to emit a report if:
  - any trial's verify.json is FAIL or missing
  - any priority-1 candidate in plan/checklist.yaml is not done/skipped
  - any metric/eval file fails signature check
  - quality gate fails on any reported trial

Prints the overrides log verbatim so bypasses are visible, not hidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.common import pinned, signing  # noqa: E402

TRIALS = REPO / "trials"
CHECKLIST = REPO / "plan" / "checklist.yaml"
OVERRIDES = REPO / "harness" / "ledger" / "overrides.log"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _trial_rows() -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    if not TRIALS.exists():
        return rows, errors
    for tdir in sorted(TRIALS.iterdir()):
        if not tdir.is_dir():
            continue
        vf = tdir / "verify.json"
        if not vf.exists():
            errors.append(f"{tdir.name}: verify.json missing")
            continue
        v = _load(vf)
        if v.get("status") != "PASS":
            errors.append(f"{tdir.name}: verify status {v.get('status')}")
            continue
        m = _load(tdir / "metrics.json")
        e = _load(tdir / "eval.json")
        for name, payload in (("metrics", m), ("eval", e)):
            ok, reason = signing.verify_signature(payload, tdir.name)
            if not ok:
                errors.append(f"{tdir.name}: {name}.json signature bad: {reason}")
        rows.append(
            {
                "trial": tdir.name,
                "method": m.get("method"),
                "bits": m.get("bit_width"),
                "weight_gb": m.get("weight_size_bytes", 0) / 1e9,
                "ttft_p50_ms": m.get("ttft_ms", {}).get("p50"),
                "ttft_p95_ms": m.get("ttft_ms", {}).get("p95"),
                "tokps_mean": m.get("decode_tokens_per_sec", {}).get("mean"),
                "lat_p50_ms": m.get("total_latency_ms", {}).get("p50"),
                "lat_p95_ms": m.get("total_latency_ms", {}).get("p95"),
                "aime_2026": e.get("accuracy"),
                "pass_gate": e.get("pass_gate"),
            }
        )
    return rows, errors


def _checklist_audit() -> list[str]:
    if not CHECKLIST.exists():
        return ["plan/checklist.yaml missing — design phase never ran"]
    cl = yaml.safe_load(CHECKLIST.read_text()) or {}
    missing = []
    for c in cl.get("candidates", []):
        if c.get("priority") == 1 and c.get("status") not in {"done", "skipped"}:
            missing.append(c.get("id"))
        if c.get("status") == "skipped" and not c.get("skip_reason"):
            missing.append(f"{c.get('id')} (skipped without reason)")
    return (
        [f"priority-1 candidate not resolved: {m}" for m in missing] if missing else []
    )


def _overrides_block() -> str:
    if not OVERRIDES.exists():
        return "_(none)_"
    return "```\n" + OVERRIDES.read_text().strip() + "\n```"


def main() -> int:
    rows, row_errors = _trial_rows()
    plan_errors = _checklist_audit()
    gate = pinned.load()["quality_gate"]["aime_2026_no_tools_min"]
    gate_fails = [r["trial"] for r in rows if r["aime_2026"] < gate]

    blocking = row_errors + plan_errors + [
        f"{t}: AIME below gate" for t in gate_fails
    ]
    if blocking:
        print("REPORT BLOCKED (R8 / R11 / R15):", file=sys.stderr)
        for e in blocking:
            print(f"  - {e}", file=sys.stderr)
        return 2

    lines = ["# Gemma-4 26B A4B-it — Quantization Session Report", ""]
    lines.append(f"Quality gate: AIME 2026 no-tools ≥ {gate:.3f}")
    lines.append("")
    lines.append("| Trial | Method | Bits | Weight GB | TTFT p50 ms | TTFT p95 ms | tok/s | Lat p50 | Lat p95 | AIME |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['trial']} | {r['method']} | {r['bits']} | "
            f"{r['weight_gb']:.2f} | {r['ttft_p50_ms']:.1f} | "
            f"{r['ttft_p95_ms']:.1f} | {r['tokps_mean']:.2f} | "
            f"{r['lat_p50_ms']:.1f} | {r['lat_p95_ms']:.1f} | "
            f"{r['aime_2026']:.3f} |"
        )
    lines.append("")
    lines.append("## Overrides log")
    lines.append(_overrides_block())
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
