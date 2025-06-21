# ML Platform Makefile

.PHONY: help install test test-unit test-integration test-coverage clean lint format format-check check fix setup-dev

# Default target
help:
	@echo "ML Platform - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install         - Install core dependencies"
	@echo "  install-dev     - Install with development tools"
	@echo "  install-cloud   - Install with cloud dependencies"
	@echo "  install-ml      - Install with ML dependencies (Phase 2)"
	@echo "  install-all     - Install all dependencies"
	@echo "  setup-dev       - Set up development environment"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-fast      - Run fast tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run code linting with Ruff and mypy"
	@echo "  format         - Format code with Ruff"
	@echo "  format-check   - Check code formatting"
	@echo "  check          - Run all code quality checks"
	@echo "  fix            - Auto-fix linting issues and format code"
	@echo ""
	@echo "Application:"
	@echo "  run            - Run the application"
	@echo "  clean          - Clean up generated files"

# Environment setup (using uv best practices)
install:
	uv sync

install-dev:
	uv sync --extra dev

install-cloud:
	uv sync --extra cloud

install-ml:
	uv sync --extra ml

install-all:
	uv sync --all-extras

setup-dev: install-dev
	uv run pre-commit install

# Testing targets
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v -m "not slow"

test-integration:
	uv run pytest tests/integration/ -v

test-coverage:
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term

test-fast:
	uv run pytest tests/unit/ -v -m "not slow" --disable-warnings -x

# Code quality (using Ruff)
lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

check: format-check lint
	@echo "✅ All checks passed!"

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

# Application
run:
	uv run python main.py

# Cleanup
clean:
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Development workflow
dev-test: check test-fast

# CI/CD targets
ci-test: test-coverage check