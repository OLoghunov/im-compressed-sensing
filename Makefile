.PHONY: help install test test-filter clean format lint typecheck all-checks

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-filter FILTER=<name> - Run tests matching filter"
	@echo "  make clean         - Clean temporary files"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Run linter (flake8)"
	@echo "  make typecheck     - Run type checker (mypy)"
	@echo "  make all-checks    - Run format + lint + typecheck"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

test-filter:
	pytest -k "$(FILTER)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "*.pyc" -delete

format:
	black imcs/ test_imcs/ main.py

lint:
	flake8 imcs/ test_imcs/ main.py --max-line-length=100 --ignore=C901,E203

typecheck:
	mypy imcs/ --ignore-missing-imports

all-checks: format lint typecheck
	@echo "✅ All checks passed!"
