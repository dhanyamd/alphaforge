# AlphaForge — common commands. Run `make help`.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install gen audit backtest report test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (editable) + dev tools
	$(PY) -m pip install -e ".[dev,viz]"

gen:  ## Generate the synthetic data lake (prices, fundamentals, alt-data)
	$(PY) -m alphaforge.cli generate-data

audit:  ## Run data-quality audit on the lake
	$(PY) -m alphaforge.cli audit

backtest:  ## Run the full end-to-end multi-factor backtest
	$(PY) -m alphaforge.cli backtest

analyze:  ## Prove the model predicts (IC + quantile spread + equity-curve plot)
	$(PY) -m alphaforge.cli analyze

compare:  ## Head-to-head: linear alpha vs gradient-boosted-tree (ML) alpha
	$(PY) -m alphaforge.cli compare-models

importance:  ## Which signals did the ML model rely on? (permutation importance)
	$(PY) -m alphaforge.cli feature-importance

report:  ## Print the latest backtest performance tearsheet
	$(PY) -m alphaforge.cli report

test:  ## Run the test suite
	pytest -q

lint:  ## Lint + format check
	ruff check src tests

clean:  ## Remove generated data + artifacts
	rm -rf data_lake/*.parquet data_lake/**/*.parquet data_lake/*.duckdb artifacts/*
