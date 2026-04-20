"""Block Edit/Write/NotebookEdit on sealed paths (R5, R7, R13).

Sealed: harness/**, configs/pinned.yaml, .claude/agents/evaluator.md
Override (loggable, bounded): ALLOW_HARNESS_EDIT=<reason> in env.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "harness" / "ledger" / "overrides.log"

SEALED_PREFIXES = ("harness/", ".claude/agents/evaluator.md")
SEALED_FILES = ("configs/pinned.yaml",)


def _is_sealed(rel: str) -> bool:
    return rel.startswith(SEALED_PREFIXES) or rel in SEALED_FILES


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    inp = payload.get("tool_input", {})
    path = inp.get("file_path") or inp.get("notebook_path") or ""
    if not path:
        return 0
    try:
        rel = str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        rel = path
    if not _is_sealed(rel):
        return 0

    reason = os.environ.get("ALLOW_HARNESS_EDIT")
    if reason:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(
                json.dumps({"ts": time.time(), "path": rel, "reason": reason}) + "\n"
            )
        return 0

    print(
        f"SEALED PATH: {rel} is harness-owned (R5/R13). "
        "Set ALLOW_HARNESS_EDIT=<reason> only if the edit is a harness fix itself.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
