.PHONY: help install lint format typecheck test test-all ingest train eval benchmark verify serve docker clean all

help:
	@echo "install    Install the package with dev dependencies"
	@echo "lint       Ruff check and format check"
	@echo "format     Apply Ruff formatting and safe fixes"
	@echo "typecheck  Mypy in strict mode"
	@echo "test       Unit tests (offline, no models)"
	@echo "test-all   Unit and integration tests"
	@echo "ingest     Build the index from EDGAR"
	@echo "train      Train the query router and the LTR / XGBoost rerankers"
	@echo "eval       Run the golden set with the quality gate"
	@echo "benchmark  Ablate retrieval arms and rerankers"
	@echo "verify     Check README claims against the generated reports"
	@echo "serve      Run the API and UI"
	@echo "all        lint, typecheck, test"

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check src tests --fix

typecheck:
	mypy src

test:
	pytest tests/unit -v --no-cov

test-all:
	pytest tests -v

ingest:
	python -m secrag.cli ingest --rebuild

train:
	python -m secrag.cli train-router
	python -m secrag.cli train-ltr
	python -m secrag.cli train-xgb

eval:
	python -m secrag.cli eval --gate

benchmark:
	python -m secrag.cli benchmark
	python scripts/update_readme.py

verify:
	python scripts/verify_claims.py

serve:
	python -m secrag.cli serve

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

all: lint typecheck test verify
