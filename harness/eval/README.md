# AIME 2026 — sealed eval runner

`aime_runner.py` is the only sanctioned evaluator for AIME 2026 numbers.
Parameters live in `configs/pinned.yaml` under `eval:` and cannot be
overridden by CLI flags (R5).

`aime_2026.json` holds the problem set — seed it once with the MathArena
release (schema: `[{id, problem, answer}]`) and commit it. The executor agent
cannot modify it (PreToolUse hook blocks writes under `harness/**`).

The quality gate is `quality_gate.aime_2026_no_tools_min = 0.853` in
`pinned.yaml`. `harness/verify/run.py` fails any trial below this floor (R15).
