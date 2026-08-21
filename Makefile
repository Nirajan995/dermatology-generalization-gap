.PHONY: help install install-full test lint data baseline features evaluate figures null all clean

help:
	@echo "install       CPU-only analysis dependencies"
	@echo "install-full  add PyTorch stack for training and feature extraction"
	@echo "test          run the data-free test suite"
	@echo "lint          ruff check"
	@echo "data          build unified dataset tables"
	@echo "baseline      fine-tune the cancer baseline (GPU)"
	@echo "features      extract frozen features (GPU, resumable)"
	@echo "evaluate      run all three analyses"
	@echo "figures       regenerate Figure 2"
	@echo "null          tone-gap null calibration"
	@echo "all           full pipeline (scripts/run_all.sh)"
	@echo "clean         remove generated artifacts and caches"

install:
	pip install -e .

install-full:
	pip install -e ".[full,dev]"

test:
	pytest -q

lint:
	ruff check src tests

data:
	python -m dermgap.datasets

baseline:
	python -m dermgap.train_baseline --epochs 15

features:
	python -m dermgap.extract_features --models all --datasets all

evaluate:
	python -m dermgap.evaluate --analysis all

figures:
	python -m dermgap.figures --granularity fine
	python -m dermgap.figures --granularity category

null:
	python -m dermgap.null_calibration

all:
	bash scripts/run_all.sh

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
	rm -f results/decomposition.json results/purity.json results/adaptation.json results/summary.md
