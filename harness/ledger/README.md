# Ledger

Append-only records the harness uses to enforce R7 / R9 / R11:

- `overrides.log` — every `ALLOW_POLICY_OVERRIDE=` / `ALLOW_HARNESS_EDIT=`
  invocation is recorded here (JSON lines: `ts, reason, command|path`). The
  report pipeline surfaces this file in full — overrides are visible, not
  hidden.
- `trials.jsonl` — written by `harness/verify/run.py` on each PASS/FAIL with
  `{trial_id, status, metrics_summary, eval_summary, git_sha, ts}`. This is
  the single source of truth for cross-trial comparisons (R9 forbids copying
  numbers out of this ledger into a different trial).

Neither file is editable by the executor agent (blocked by PreToolUse hook).
