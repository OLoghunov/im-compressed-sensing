.PHONY: help install test test-filter clean format lint

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-filter FILTER=<name> - Run tests matching filter"
	@echo "  make clean         - Clean temporary files"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Run linter (flake8)"

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
	black imcs test_imcs

lint:
	flake8 imcs test_imcs

