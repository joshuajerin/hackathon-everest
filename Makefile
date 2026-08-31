.PHONY: setup setup-isaac-unit lint test test-isaac-unit smoke verify verify-all build clean

setup:
	uv sync --group dev --locked

setup-isaac-unit:
	uv sync --extra gpu --group dev --locked
	uv pip install --python .venv/bin/python pyarrow==21.0.0 zarr==3.1.5

lint:
	uv run ruff check .

test:
	uv run pytest -q

test-isaac-unit: setup-isaac-unit
	.venv/bin/python -m pytest -q isaaclab_ext/tests/unit

smoke:
	uv run everest pipeline --config configs/smoke.yaml --out artifacts/smoke

verify: lint test smoke

verify-all: verify test-isaac-unit

build:
	uv build

clean:
	rm -rf build dist .pytest_cache .ruff_cache artifacts/smoke
