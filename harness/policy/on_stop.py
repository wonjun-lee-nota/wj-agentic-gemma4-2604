"""PostToolUse / Stop hook — auto-run verify for any trial that has fresh
artifacts but no verify.json yet, and fail the turn if any verify is FAIL
(R8).

Also enforces R11: if plan/checklist.yaml has unchecked priority-1 items and
the session is stopping, print a reminder (non-blocking).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TRIALS = REPO / "trials"
CHECKLIST = REPO / "plan" / "checklist.yaml"


def _needs_verify(tdir: Path) -> bool:
    has_intent = (tdir / "intent.yaml").exists()
    has_metrics = (tdir / "metrics.json").exists()
    has_eval = (tdir / "eval.json").exists()
    verify = tdir / "verify.json"
    if not (has_intent and (has_metrics or has_eval)):
        return False
    if not verify.exists():
        return True
    # Re-verify if artifacts changed after verify.json
    v_mtime = verify.stat().st_mtime
    for fname in ("intent.yaml", "metrics.json", "eval.json", "config.json"):
        p = tdir / fname
        if p.exists() and p.stat().st_mtime > v_mtime:
            return True
    return False


def main() -> int:
    if not TRIALS.exists():
        return 0
    failures: list[str] = []
    for tdir in TRIALS.iterdir():
        if not tdir.is_dir():
            continue
        if _needs_verify(tdir):
            res = subprocess.run(
                [sys.executable, str(REPO / "harness" / "verify" / "run.py"),
                 "--trial", tdir.name],
                cwd=str(REPO),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                failures.append(tdir.name)
        else:
            vf = tdir / "verify.json"
            if vf.exists():
                try:
                    if json.loads(vf.read_text()).get("status") != "PASS":
                        failures.append(tdir.name)
                except Exception:
                    failures.append(tdir.name)

    if CHECKLIST.exists():
        try:
            cl = yaml.safe_load(CHECKLIST.read_text()) or {}
            unrun_p1 = [
                c["id"]
                for c in cl.get("candidates", [])
                if c.get("priority") == 1 and c.get("status") not in {"done", "skipped"}
            ]
            if unrun_p1:
                print(
                    f"[R11] priority-1 candidates pending: {unrun_p1}",
                    file=sys.stderr,
                )
        except Exception:
            pass

    if failures:
        print(
            f"[R8] verify FAIL for trials: {failures}. "
            "Fix the trial artifacts or mark skipped in plan/checklist.yaml.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
