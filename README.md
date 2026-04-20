# wj-agentic-gemma4-2604

Agentic quantization harness for `google/gemma-4-26B-A4B-it` on a single
NVIDIA RTX 3090. The agent explores the quantization search space; the
harness owns measurement, evaluation, verification, and policy.

Start at [`CLAUDE.md`](./CLAUDE.md) for the session contract and at
[`docs/harness_design.md`](./docs/harness_design.md) for how the layout maps
onto the R5–R15 requirements from the Confluence spec.

## Quick map

| Area | Path |
|---|---|
| Session contract | `CLAUDE.md` |
| Pinned measurement / eval params | `configs/pinned.yaml` |
| Plan & schemas | `plan/` |
| Sealed harness | `harness/` (metrics, eval, verify, policy, ledger) |
| Subagents | `.claude/agents/{design,executor,evaluator}.md` |
| Hooks / permissions | `.claude/settings.json` |
| Trials | `trials/<id>/` |
| Report | `make report` (→ `scripts/make_report.py`) |
