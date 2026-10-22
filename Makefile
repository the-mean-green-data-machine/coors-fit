# Coors Fit pipeline. `make all` rebuilds everything from raw Statcast pulls.
# Requires uv (https://docs.astral.sh/uv/); it installs the pinned environment on first use.

UV ?= uv
RUN := $(UV) run --frozen

.DEFAULT_GOAL := help
.PHONY: help setup openmp data features targets gate models eval report all test lint requirements

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

features:  ## Phase 4: non-Coors traits from prior seasons
	@echo "features: not implemented yet (Phase 4)"; exit 1

targets:  ## Phase 2: hitter-season Coors residuals, shrinkage and diagnostics
	$(RUN) python -m coorsfit.targets
	$(RUN) python -m coorsfit.targets_report > /dev/null

models:  ## Phase 4: baseline B0 and candidates M1, M2
	@echo "models: not implemented yet (Phase 4)"; exit 1

gate:  ## Phase 3: is there a stable Coors skill? (between-hitter SD with bootstrap)
	$(RUN) python -m coorsfit.gate > /dev/null

eval: gate  ## Phases 3-6: gate, out-of-sample lift, mechanism and robustness checks
	@echo "eval: Phases 4-6 not implemented yet"; exit 1

report:  ## Phase 7: figures and tables for the README and memo
	@echo "report: not implemented yet (Phase 7)"; exit 1

all: setup data targets features models eval report  ## Rebuild everything from raw pulls

test:  ## Run the test suite
	$(RUN) pytest -q

lint:  ## Check style with ruff
	$(RUN) ruff check src tests

requirements:  ## Regenerate requirements.txt from uv.lock (for pip users)
	$(UV) export --frozen --no-hashes --no-emit-project -o requirements.txt
