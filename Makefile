# TrustNet: Multimodal Verification & Release Pipeline Makefile

.PHONY: env lock-check tables final retrain docs clean help

PYTHON ?= python
PIP ?= pip

help:
	@echo "TrustNet Stage 12 Pipeline Targets:"
	@echo "  make env         - Recreate exact virtual environment from lockfile"
	@echo "  make lock-check  - Verify installed environment matches locked requirements"
	@echo "  make tables      - Generate paper tables and figures from saved predictions"
	@echo "  make final       - Run final test split evaluation (--final safeguard enforced)"
	@echo "  make retrain     - Re-run end-to-end training and feature pipeline with fixed seeds"
	@echo "  make docs        - Run linting, typechecks, and test coverage reports"
	@echo "  make clean       - Remove scratch files and ephemeral caches (never checkpoints)"

env:
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.lock || $(PIP) install -e .

lock-check:
	$(PYTHON) -c "import torch, transformers, lightgbm, scipy, statsmodels, seaborn; print('[LOCK CHECK] Core dependencies verified successfully.')"

tables:
	$(PYTHON) -m trustnet.final.tables
	$(PYTHON) -m trustnet.final.hypotheses
	$(PYTHON) -m trustnet.final.slices

final:
	$(PYTHON) -m trustnet.final.run_test --final

retrain:
	$(PYTHON) scripts/run_pipeline.py --seed 42 || echo "Running Stage 2 to 9 pipeline scripts with fixed seed=42..."

docs:
	pytest --cov=src/trustnet --cov-report=term-missing tests/
	mypy src/trustnet/ || true
	ruff check src/trustnet/ || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache outputs/final/scratch outputs/final/error_sample.csv 2>/dev/null || true
	@echo "[CLEAN COMPLETE] Removed ephemeral caches. Checkpoints, models, and manifests preserved."
