.PHONY: help install install-gui clean format lint typecheck all-checks

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies (pip)"
	@echo "  make install-gui   - pip + подсказка по Tk для GUI (macOS: brew install python-tk)"
	@echo "  make clean         - Clean temporary files"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Run linter (flake8)"
	@echo "  make typecheck     - Run type checker (mypy)"
	@echo "  make all-checks    - Run format + lint + typecheck"

install:
	pip install -r requirements.txt
	pip install -e .

install-gui: install
	@echo ""
	@echo "Если GUI падает на import tkinter / _tkinter: Tk не из pip."
	@echo "  macOS (Homebrew):  brew install python-tk"
	@echo "  Затем пересоздайте venv: python3 -m venv venv && source venv/bin/activate && make install"
	@echo ""
	@python3 -c "import tkinter; print('tkinter: OK')" 2>/dev/null || \
		(echo "tkinter недоступен — см. выше."; exit 1)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "*.pyc" -delete

format:
	black imcs/ run.py main.py

lint:
	flake8 imcs/ run.py main.py --max-line-length=100 --ignore=C901,E203,W503

typecheck:
	mypy imcs/ --ignore-missing-imports

all-checks: format lint typecheck
	@echo "✅ All checks passed!"
