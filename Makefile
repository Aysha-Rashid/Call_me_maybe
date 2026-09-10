install:
	uv sync

run:
	uv run python -m src \
	--input data/input/function_calling_tests.json \
	--functions_definition data/input/functions_definition.json \
	--output data/output/function_calls.json

debug:
	uv run python -m pdb -c continue -m src

clean:
	rm __pycache__ .mypy_cache src/__pycache__

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
