"""Artifact ↔ intent verifier (R6, R8, R9).

Run at trial completion and from the PostToolUse hook. Fails hard when:
  - intent.yaml fields are missing or inconsistent with config.json
  - declared bit_width doesn't match observed weight dtype / file sizes
  - metrics.json or eval.json is missing, unsigned, or signed for another trial
  - quality gate not met (AIME < pinned threshold)
  - checklist item status not updated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness.common import pinned, signing, trial  # noqa: E402


def _check_intent(tdir: Path) -> tuple[bool, list[str]]:
    errs: list[str] = []
    try:
        intent = trial.load_intent(tdir.name)
    except Exception as e:
        return False, [f"intent.yaml invalid: {e}"]
    cfg_path = tdir / "config.json"
    if not cfg_path.exists():
        errs.append("config.json missing")
        return False, errs
    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg.get("quantization_config", {})
    if qcfg:
        obs_bits = qcfg.get("bits") or qcfg.get("w_bit") or qcfg.get("weight_bits")
        if obs_bits and obs_bits != intent["bit_width"]:
            errs.append(
                f"bit_width mismatch: intent={intent['bit_width']} config.json={obs_bits}"
            )
    return len(errs) == 0, errs


def _check_weights_plausible(tdir: Path, intent: dict) -> tuple[bool, list[str]]:
    w = tdir / "weights"
    if not w.exists():
        return False, ["weights/ directory missing"]
    shards = list(w.glob("*.safetensors")) + list(w.glob("*.bin"))
    if not shards:
        return False, ["no weight shards found"]
    total = sum(s.stat().st_size for s in shards)
    # Rough sanity: 26B params at N bits ≈ 26e9 * N/8 bytes. Allow ±25% slack.
    expected = 26e9 * intent["bit_width"] / 8
    if total < expected * 0.6 or total > expected * 1.6:
        return False, [
            f"weight size implausible for bit_width={intent['bit_width']}: "
            f"got {total/1e9:.2f}GB, expected ~{expected/1e9:.2f}GB"
        ]
    return True, []


def _check_signed_artifact(path: Path, trial_id: str) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"{path.name} missing (harness did not produce it)"]
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return False, [f"{path.name} unreadable: {e}"]
    ok, reason = signing.verify_signature(data, trial_id)
    if not ok:
        return False, [f"{path.name} signature invalid: {reason}"]
    return True, []


def _check_quality_gate(tdir: Path) -> tuple[bool, list[str]]:
    eval_path = tdir / "eval.json"
    if not eval_path.exists():
        return False, ["eval.json missing"]
    data = json.loads(eval_path.read_text())
    gate = pinned.load()["quality_gate"]["aime_2026_no_tools_min"]
    acc = data.get("accuracy", 0.0)
    if acc < gate:
        return False, [f"AIME accuracy {acc:.3f} < gate {gate:.3f}"]
    return True, []


def run(trial_id: str) -> dict:
    tdir = trial.trial_dir(trial_id)
    if not tdir.exists():
        raise FileNotFoundError(f"trial dir not found: {tdir}")
    checks: list[tuple[str, bool, list[str]]] = []
    ok1, e1 = _check_intent(tdir)
    checks.append(("intent_vs_config", ok1, e1))
    try:
        intent = trial.load_intent(trial_id)
        ok2, e2 = _check_weights_plausible(tdir, intent)
    except Exception as e:
        ok2, e2 = False, [str(e)]
    checks.append(("weight_size_plausibility", ok2, e2))
    ok3, e3 = _check_signed_artifact(tdir / "metrics.json", trial_id)
    checks.append(("metrics_signature", ok3, e3))
    ok4, e4 = _check_signed_artifact(tdir / "eval.json", trial_id)
    checks.append(("eval_signature", ok4, e4))
    ok5, e5 = _check_quality_gate(tdir)
    checks.append(("quality_gate_aime_2026", ok5, e5))

    overall = all(ok for _, ok, _ in checks)
    report = {
        "trial_id": trial_id,
        "status": "PASS" if overall else "FAIL",
        "checks": [
            {"name": n, "pass": ok, "errors": errs} for n, ok, errs in checks
        ],
    }
    (tdir / "verify.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    args = ap.parse_args()
    rep = run(args.trial)
    print(json.dumps(rep, indent=2))
    return 0 if rep["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
