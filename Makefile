.PHONY: install test lint

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src tests
