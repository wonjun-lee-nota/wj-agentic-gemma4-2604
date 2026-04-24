# Token Usage Report — Session 2026-04-21 ~ 04-24

## Summary

| Item | Value |
|------|-------|
| **Total cost** | **$51.40** |
| Session duration | 434 min (~7.2 hours) |
| API duration | 76 min |
| Input tokens | 26,981 |
| Output tokens | 215,398 |
| Cache creation tokens | 670,986 |
| Cache read tokens | 85,553,204 |
| Lines added | 1,221 |
| Lines removed | 59 |

## Model Breakdown

| Model | Input | Output | Cache Create | Cache Read | Cost |
|-------|-------|--------|-------------|------------|------|
| Claude Opus 4.6 (1M) | 5,147 | 208,194 | 493,918 | 85,404,133 | $51.02 (99.3%) |
| Claude Haiku 4.5 | 21,834 | 7,204 | 177,068 | 149,071 | $0.38 (0.7%) |

## Scope

이번 세션에서 수행한 전체 작업:

1. **환경 설정**: GitHub MCP 설치, 레포 분석, CLAUDE.md 파악
2. **Phase 1 (t001-t003)**: 표준 PTQ 시도 (GPTQ, NF4, unfused) — 모두 실패
3. **Phase 2 (t004-t006)**: 대안 방법 탐색 (vLLM, torchao, GGUF) — 2개 성공
4. **Phase 3 (t007-t009)**: layer-wise 최적화 (imatrix, attn=Q8, Q5_K_M)
5. **Phase 4 (t010-t014)**: 정밀 mixed precision 설계 (5개 전략)
6. **AIME 2026 평가**: t006 (77.5% FAIL), t007 (89.2% PASS), t008 (88.3% PASS), t011 (중단)
7. **보고서 작성 및 레포 push**: 4회 커밋

## Cost Efficiency

| Metric | Value |
|--------|-------|
| Cost per trial (14 trials) | $3.67 |
| Cost per AIME eval (4 evals) | $12.85 |
| Cost per PASS trial (2 trials) | $25.70 |
| Cost per commit (4 commits) | $12.85 |

---

*Generated from Claude Code session metrics*
