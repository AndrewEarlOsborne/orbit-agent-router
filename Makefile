.PHONY: help install-dev clean format format-check lint lint-fix typecheck check test all

PYTHON := python3
PIP := $(PYTHON) -m pip
PACKAGE_DIR := orbit
TEST_DIR := tests
ALL_DIRS := $(PACKAGE_DIR) $(TEST_DIR)

help:
	@echo "Available targets:"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make format         - Format code with ruff"
	@echo "  make format-check   - Check code formatting without changes"
	@echo "  make lint           - Lint code with ruff"
	@echo "  make lint-fix       - Lint and auto-fix issues with ruff"
	@echo "  make typecheck      - Type check with mypy"
	@echo "  make test           - Run tests with pytest"
	@echo "  make clean          - Remove cache and build artifacts"
	@echo "  make all            - Run all checks and tests"

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -e ".[dev,mcp,langchain]"

clean:
	@echo "Cleaning cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup complete."

format:
	@echo "Formatting code with ruff..."
	$(PYTHON) -m ruff format $(ALL_DIRS)
	@echo "Formatting complete."

format-check:
	@echo "Checking code formatting..."
	$(PYTHON) -m ruff format --check --diff $(ALL_DIRS)

lint:
	@echo "Linting code with ruff..."
	$(PYTHON) -m ruff check --fix $(ALL_DIRS)

lint-check:
	@echo "Linting and fixing code with ruff..."
	$(PYTHON) -m ruff check $(ALL_DIRS)

typecheck:
	@echo "Type checking with mypy..."
	$(PYTHON) -m mypy $(PACKAGE_DIR)
	@echo "Type checking tests..."
	$(PYTHON) -m mypy $(TEST_DIR) --disable-error-code=import-untyped --no-warn-unused-ignores || true

check: format-check lint-check typecheck
	@echo "All checks passed!"

fix: lint format
	@echo "All auto-fixes applied"

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TEST_DIR)

all: check test
	@echo "All checks and tests completed successfully!"
