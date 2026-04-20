# AIME 2026 — sealed eval runner

`aime_runner.py` is the only sanctioned evaluator for AIME 2026. All
parameters live in `configs/pinned.yaml:eval` and cannot be overridden by
CLI flags (R5).

## Protocol (frozen)

| Field | Value |
|---|---|
| Dataset | `MathArena/aime_2026` (AIME-I + AIME-II, 30 problems) |
| `n_problems` | 30 |
| `sample_n` | 4 |
| `temperature` | 0.6 |
| `top_p` | 0.95 |
| `top_k` | unset |
| `max_tokens` | 64000 |
| `ctx_size` | 72000 |
| `chat_template` | Gemma 4 default render (thinking auto-included) |
| `strict_parsing` | false |
| `extraction` | (1) balanced-brace `\boxed{N}` / `\fbox{N}`, parse as int ∈ [0, 999]. (2) fallback (strict_parsing=false): last `\b\d+\b` in text ∈ [0, 999]. |
| `scoring` | `correct / (n_problems × sample_n)` (per-sample pass rate) |

The user-turn prompt (pinned):

```
{problem}

Put your final answer within \boxed{}.
The answer is an integer between 0 and 999 inclusive.
```

## Dataset seeding

`aime_2026.json` is the **authoritative frozen snapshot**. Seed it once:

```
pip install datasets
python scripts/fetch_aime_2026.py
```

The runner has no network side effect at trial time; the frozen snapshot is
what the verifier compares every trial against. Re-freezing is a deliberate
harness change (requires `ALLOW_HARNESS_EDIT=<reason>`) because it
invalidates cross-session comparability.

## Quality gate

`configs/pinned.yaml:quality_gate.aime_2026_no_tools_min = 0.853`. The
verifier (`harness/verify/run.py`) fails any trial with
`accuracy < 0.853` — where `accuracy = total_correct / 120` (R15).
