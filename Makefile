PY := python3

.PHONY: help report verify-all clean-trials

help:
	@echo "Targets:"
	@echo "  make report       Aggregate PASS trials into a session report"
	@echo "  make verify-all   Re-run verify for every trials/<id>"
	@echo "  trial=<id> make metrics | eval | verify"

metrics:
	@test -n "$(trial)" || (echo "usage: trial=<id> make metrics"; exit 2)
	CUDA_VISIBLE_DEVICES=0 $(PY) -m harness.metrics.bench --trial $(trial)

eval:
	@test -n "$(trial)" || (echo "usage: trial=<id> make eval"; exit 2)
	CUDA_VISIBLE_DEVICES=0 $(PY) -m harness.eval.aime_runner --trial $(trial)

verify:
	@test -n "$(trial)" || (echo "usage: trial=<id> make verify"; exit 2)
	$(PY) -m harness.verify.run --trial $(trial)

verify-all:
	@for d in trials/*/; do \
	  id=$$(basename $$d); \
	  echo "--- verify $$id ---"; \
	  $(PY) -m harness.verify.run --trial $$id || exit 2; \
	done

report:
	$(PY) scripts/make_report.py
