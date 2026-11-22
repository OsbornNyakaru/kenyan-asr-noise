PY ?= .venv/bin/python

.PHONY: help setup test toy smoke status submit pull report hooks check-public data analysis

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-13s %s\n", $$1, $$2}'

setup:  ## Create .venv with the packages.
	uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt

test:  ## Run the tests (no data or models needed).
	$(PY) -m pytest -q tests

toy:  ## Write synthetic clips and noise to data/raw/ (refuses to overwrite real data).
	$(PY) scripts/make_toy_data.py

smoke: toy  ## End-to-end check of the remote path on toy data with the oracle system (no model download).
	$(PY) autoresearch/remote.py EXP-001 --clips 5 \
	  --set 'systems.kik={"family":"oracle"}' --set 'systems.swh={"family":"oracle"}' --set 'systems.eng={"family":"oracle"}'

status:  ## List experiments and where each one is.
	$(PY) scripts/exp.py status

submit:  ## Queue specs on Hugging Face for molab (commit and push first).
	$(PY) scripts/exp.py submit

pull:  ## Fetch molab results into experiments/runs/ and rebuild RESULTS.md.
	$(PY) scripts/exp.py pull

report:  ## Rebuild experiments/RESULTS.md from results.jsonl.
	$(PY) scripts/exp.py report

data:  ## Build data/raw/ and push it to the PRIVATE HF data repo. Set SOURCES and NOISE, see scripts/build_data.py.
	$(PY) scripts/build_data.py $(SOURCES) --noise $(NOISE) --push

analysis:  ## Open the analysis notebook.
	$(PY) -m marimo edit notebooks/analysis.py

hooks:  ## Install the pre-commit check that keeps recordings out of this public repository.
	printf '#!/bin/sh\nexec .venv/bin/python scripts/check_public.py\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

check-public:  ## Scan every tracked file for recordings and data files.
	$(PY) scripts/check_public.py --all
