# Gemma-4 26B A4B-it Quantization Harness

This repository is a **sealed harness** for exploring quantization of
`google/gemma-4-26B-A4B-it` onto a single NVIDIA RTX 3090. The agent's job
is to explore quantization configurations. The harness owns measurement,
evaluation, verification, and policy — those are **not** part of the search
space.

## Target

| Item | Value |
|---|---|
| Model | `google/gemma-4-26B-A4B-it` (MoE, ~26 B params) |
| Deploy HW | 1 × NVIDIA RTX 3090 (24 GB) |
| Quant workers | up to 4 × RTX 3090 (calibration / conversion only) |
| Quality gate | AIME 2026 no-tools **≥ 85.3 %** |
| Perf reported | TTFT, tokens/sec (128→128, batch=1), p50/p95 latency, weight_size |
| Measurement policy | **20 runs, 5 warmup**, `CUDA_VISIBLE_DEVICES=0` |

## How to work in this repo

1. **Plan first.** The `design` subagent fills `plan/checklist.yaml` (schema
   in `plan/checklist.schema.yaml`) with candidate configs, priorities, and
   phases. Agents must not start a trial that does not appear as a candidate
   in the checklist.
2. **One trial = one directory** under `trials/<id>/`. Required inputs:
   - `intent.yaml` (schema: `plan/trial_intent.schema.yaml`)
   - `weights/` with quantized model files
   - `config.json` (transformers-compatible, including `quantization_config`)
3. **Measurement & eval are harness-owned.** Run only:
   ```
   CUDA_VISIBLE_DEVICES=0 python -m harness.metrics.bench --trial <id>
   CUDA_VISIBLE_DEVICES=0 python -m harness.eval.aime_runner --trial <id>
   ```
   These emit HMAC-signed `metrics.json` / `eval.json` (R5, R9).
4. **Verify before moving on.**
   ```
   python -m harness.verify.run --trial <id>
   ```
   Writes `verify.json`; `status: FAIL` blocks downstream reporting (R6, R8).
5. **Report only at the end.**
   ```
   make report
   ```
   Enumerates all PASS trials, surfaces unrun priority-1 candidates,
   appends override-ledger contents.

## Hard rules (R7, non-overridable)

- **No pre-quantized checkpoint downloads.** Quantization must be produced
  locally from the FP16 / BF16 weights. `huggingface-cli download` of any
  `*-gptq-*`, `*-awq-*`, `*-int4-*`, `*-fp8-*`, etc. is blocked.
- **No `model.generate()` for latency/throughput measurement.** Use
  `harness/metrics/bench.py`. `.generate()` is allowed only from
  `harness/eval/aime_runner.py`.
- **Measurement is single-card.** `CUDA_VISIBLE_DEVICES=0` is enforced on any
  invocation of `harness/metrics/bench.py` and `harness/eval/aime_runner.py`.
- **Cross-trial metric copies are forbidden.** Every number file is
  HMAC-signed for its own `trial_id`; copying a signed file into another
  trial fails verification.

## Soft rules (overridable with logged reason)

- Editing harness files requires `ALLOW_HARNESS_EDIT=<reason>` (logged to
  `harness/ledger/overrides.log`). Use this only when fixing harness bugs,
  not to relax the protocol for a convenient result.
- Policy violations outside measurement (e.g., novel dataset downloads) can
  be overridden with `ALLOW_POLICY_OVERRIDE=<reason>` inline in the bash
  command.

## Allowed quantization methods

Listed in `configs/pinned.yaml → allowed_methods`. Anything outside this
list must enter `plan/checklist.yaml` as an explicit `experimental` entry
with rationale and must pass design-agent review.

## Subagents

- `.claude/agents/design.md` — proposes candidates, writes `plan/checklist.yaml`.
- `.claude/agents/executor.md` — runs trials, writes `trials/<id>/*`.
- `.claude/agents/evaluator.md` — read-only over trials + ledger; critiques
  coverage, confirms verifier outputs, drafts report.

The executor cannot edit the evaluator's prompt, the harness, or pinned
configs — those paths are sealed by `.claude/settings.json` hooks.
