# Coors Fit pipeline. `make all` rebuilds everything from raw Statcast pulls.
# Requires uv (https://docs.astral.sh/uv/); it installs the pinned environment on first use.

UV ?= uv
RUN := $(UV) run --frozen

.DEFAULT_GOAL := help
.PHONY: help setup openmp data features targets gate models mechanisms robustness eval report all test lint requirements

help:  ## List targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

setup:  ## Install the pinned environment, plus the OpenMP runtime LightGBM needs
	$(UV) sync --frozen
	@$(RUN) python -c "import lightgbm" 2>/dev/null || $(MAKE) --no-print-directory openmp

# LightGBM's wheel links against a system OpenMP runtime that pip cannot install.
# Linux usually ships libgomp; macOS needs Homebrew's libomp.
openmp:
	@if [ "$$(uname)" = Darwin ] && command -v brew >/dev/null; then \
		echo "LightGBM needs the OpenMP runtime: running 'brew install libomp'"; \
		brew install libomp; \
	else \
		echo "LightGBM cannot load an OpenMP runtime. Install one, then rerun 'make setup':"; \
		echo "  macOS: brew install libomp    Debian/Ubuntu: apt-get install libgomp1"; \
		exit 1; \
	fi
	@$(RUN) python -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'loads')"

data:  ## Phase 1: pull Statcast (cached, resumable) and build the PA table
	$(RUN) python -m coorsfit.ingest
	$(RUN) python -m coorsfit.clean

features:  ## Phase 4: non-Coors traits from prior seasons (leakage-safe)
	$(RUN) python -m coorsfit.features

targets:  ## Phase 2: hitter-season Coors residuals, shrinkage and diagnostics
	$(RUN) python -m coorsfit.targets
	$(RUN) python -m coorsfit.targets_report > /dev/null

models: gate  ## Phase 4: B0, B0q, M1, M2; out-of-sample lift, confirmatory tests, transfer test
	$(RUN) python -m coorsfit.evaluate > /dev/null

gate:  ## Phase 3: is there a stable Coors skill? (between-hitter SD with bootstrap)
	$(RUN) python -m coorsfit.gate > /dev/null

mechanisms: models  ## Phase 5: mechanism checks for H1 to H3
	$(RUN) python -m coorsfit.mechanisms > /dev/null

robustness: models  ## Phase 6: confirmatory tests and headline lift without 2020 and at a 400-PA trait minimum
	$(RUN) python -m coorsfit.robustness > /dev/null

eval: mechanisms robustness  ## Phases 3-6: gate, out-of-sample lift, mechanism and robustness checks

report:  ## Phase 7: figures and tables for the README and memo
	@echo "report: not implemented yet (Phase 7)"; exit 1

all: setup data targets features models eval report  ## Rebuild everything from raw pulls

test:  ## Run the test suite
	$(RUN) pytest -q

lint:  ## Check style with ruff
	$(RUN) ruff check src tests

requirements:  ## Regenerate requirements.txt from uv.lock (for pip users)
	$(UV) export --frozen --no-hashes --no-emit-project -o requirements.txt
