"""Execution-layer policy (R7).

Invoked as a PreToolUse hook for Bash. Reads the tool input from stdin as JSON,
inspects the command, exits non-zero to block forbidden operations:

  - downloading pre-quantized checkpoints
  - inline python scripts calling model.generate() for latency/throughput
  - measurement runs not pinned to single-card (CUDA_VISIBLE_DEVICES != "0")
  - edits to harness/** or configs/pinned.yaml (via shell redirects / sed)

Override: the agent can prefix with `ALLOW_POLICY_OVERRIDE=<reason>` and the
reason is appended to harness/ledger/overrides.log. Override on measurement
(.generate, multi-card) is still rejected — those are non-overridable.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "harness" / "ledger" / "overrides.log"

PRE_QUANTIZED = re.compile(
    r"(huggingface-cli\s+download|hf\s+download|git\s+clone\s+\S*huggingface\.co/)"
    r".*(gptq|awq|int4|int8|fp8|bnb|quantized|-q4|-q8)",
    re.IGNORECASE,
)

GENERATE_IN_INLINE_SCRIPT = re.compile(
    r"python3?\s+-c\s+['\"].*\.generate\(", re.DOTALL
)

SHELL_EDIT_OF_HARNESS = re.compile(
    r"(>\s*|>>\s*|tee\s+|sed\s+-i\s+\S+\s+)"
    r"(harness/|configs/pinned\.yaml)",
)

NON_OVERRIDABLE = ("measurement with .generate", "multi-card measurement")


def _log_override(reason: str, command: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(
            json.dumps(
                {"ts": time.time(), "reason": reason, "command": command}
            )
            + "\n"
        )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # malformed input — let Claude Code handle
    cmd = payload.get("tool_input", {}).get("command", "")
    if not cmd:
        return 0

    override = re.search(r"ALLOW_POLICY_OVERRIDE=([^\s]+)", cmd)
    override_reason = override.group(1) if override else None

    violations: list[str] = []

    if PRE_QUANTIZED.search(cmd):
        violations.append("pre-quantized checkpoint download forbidden (R7)")

    if GENERATE_IN_INLINE_SCRIPT.search(cmd):
        violations.append(
            "inline .generate() forbidden for measurement — use harness/metrics/bench.py"
        )

    if SHELL_EDIT_OF_HARNESS.search(cmd):
        violations.append("shell write into harness/ or configs/pinned.yaml forbidden")

    # Measurement entry points require CUDA_VISIBLE_DEVICES=0. Match both the
    # path form (harness/metrics/bench.py) and the module form
    # (-m harness.metrics.bench), same for aime_runner.
    measurement_patterns = (
        "harness/metrics/bench.py",
        "harness.metrics.bench",
        "harness/eval/aime_runner.py",
        "harness.eval.aime_runner",
    )
    if any(p in cmd for p in measurement_patterns):
        has_single_card = re.search(
            r"(^|\s)CUDA_VISIBLE_DEVICES=0(\s|$)", cmd
        )
        if not has_single_card:
            violations.append(
                "measurement must run with CUDA_VISIBLE_DEVICES=0 (single 3090)"
            )

    if not violations:
        return 0

    non_overridable_hit = any(
        "generate()" in v or "multi-card" in v or "measurement" in v
        for v in violations
    )

    if override_reason and not non_overridable_hit:
        _log_override(override_reason, cmd)
        return 0

    print("POLICY BLOCK:\n  - " + "\n  - ".join(violations), file=sys.stderr)
    print(
        "\nIf genuinely intended, re-run with ALLOW_POLICY_OVERRIDE=<reason> "
        "(not allowed for measurement-path violations).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
