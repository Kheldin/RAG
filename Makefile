.PHONY: install run debug clean lint lint-strict help

install:
	@echo "Installing dependencies with uv..."
	uv sync

run:
	uv run python3 src/main.py

debug:
	@echo "Running in debug mode..."
	uv run python -m pdb -m src

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Clean complete."

lint:
	@echo "Running flake8..."
	flake8 .
	@echo "Running mypy..."
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running flake8..."
	flake8 src
	@echo "Running mypy (strict mode)..."
	mypy src --strict