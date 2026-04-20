# Harness design — mapping to the R5–R15 requirements

The Confluence spec lists eight requirements the harness must satisfy. Each
one maps to a specific file or mechanism below.

| Req | Intent | Where it lives |
|---|---|---|
| **R5** — measurement is harness-owned, params pinned | `harness/metrics/bench.py`, `harness/eval/aime_runner.py`, params in `configs/pinned.yaml`. Edits to these paths are blocked by `.claude/settings.json` + `harness/policy/check_edit.py`. |
| **R6** — artifact ↔ intent verifier | `harness/verify/run.py` cross-checks `trials/<id>/intent.yaml` against `config.json`, weight sizes, and bit width. Writes `verify.json`; non-PASS blocks downstream. |
| **R7** — allowed/forbidden paths as execution policy | `harness/policy/check_bash.py` (PreToolUse on Bash) blocks pre-quantized downloads, ad-hoc `.generate()` measurement, and multi-GPU measurement. Override for non-measurement-path violations requires `ALLOW_POLICY_OVERRIDE=<reason>`, which is appended to `harness/ledger/overrides.log`. |
| **R8** — trial → auto-verify → block pipeline | `harness/policy/on_stop.py` (Stop hook) runs verify on any trial with fresh artifacts. `scripts/make_report.py` refuses to emit a report if any verify FAIL or any priority-1 checklist item is unresolved. |
| **R9** — measured values only | `harness/common/signing.py` HMAC-signs every `metrics.json` / `eval.json` with `(trial_id, engine, timestamp, git_sha, payload)`. The key lives in `harness/common/.harness_key` (git-ignored, 0600). The verifier rejects signatures whose `trial_id` does not match the directory — this is what catches cross-trial copies. |
| **R11** — plan-vs-execution tracking | `plan/checklist.yaml` is populated by the `design` subagent. `on_stop.py` flags unrun priority-1 candidates; `make_report.py` refuses to emit unless each is `done` or `skipped` (with `skip_reason`). |
| **R13** — 3-role architecture | `.claude/agents/{design,executor,evaluator}.md`. Hooks make the split enforceable: design has no Bash, evaluator has read-only tools, executor cannot edit sealed paths, no agent can edit `.claude/agents/evaluator.md`. |
| **R15** — quality-degradation budget | `configs/pinned.yaml → quality_gate.aime_2026_no_tools_min = 0.853`. The verifier fails any trial below this floor, so no number with AIME < 85.3 % can become a "pass". `degradation_budget_abs` is a hook for the design agent to register a baseline-tied drop budget. |

## The executor's feedback loop

```
design → plan/checklist.yaml
   │
   ▼
executor picks candidate → trials/<id>/intent.yaml + weights/ + config.json
   │
   ▼
CUDA_VISIBLE_DEVICES=0 python -m harness.metrics.bench  --trial <id>   (signs metrics.json)
CUDA_VISIBLE_DEVICES=0 python -m harness.eval.aime_runner --trial <id> (signs eval.json)
python -m harness.verify.run --trial <id>                              (writes verify.json)
   │
   ▼
Stop hook re-verifies and blocks the turn on FAIL
   │
   ▼
evaluator reads verify.json, eval.json, metrics.json, overrides.log
   │
   ▼
make report (only if every PASS gate + checklist closure + signature check clears)
```

## Non-goals of this harness

- It does **not** sandbox the quantization step itself. The agent is free to
  choose algorithms, calibration sets, layer targets — that's the search
  space.
- It does **not** attempt to rewrite failing trials. A FAIL stays FAIL; the
  executor must either fix inputs and retry, or mark the candidate
  `skipped` with a reason.
- It does **not** gate on non-AIME metrics. TTFT / tok-per-sec / weight-size
  are reported but not pass/fail — the quality gate is AIME 2026 ≥ 85.3 %,
  with perf reported for ranking.

## When to update `configs/pinned.yaml`

You shouldn't, during a session. If the pinned measurement protocol is
genuinely wrong (e.g., the AIME extractor has a bug), fix it in a separate
PR with `ALLOW_HARNESS_EDIT=<reason>` — every change to pinned params
invalidates cross-session comparability and must be explicit.
