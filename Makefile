.PHONY: gate test

test:
	python -m pytest -q

gate:
	ruff format --check .
	ruff check .
	python -m compileall src tests
	python -m pytest
	mypy --follow-imports=skip src/
	pip-audit --local
	bandit -r src -c pyproject.toml
