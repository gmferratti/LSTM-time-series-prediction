install:
	python -m pip install --upgrade pip
	py -3.12 -m venv .venv
	source .venv/Scripts/activate
	python -m pip install --upgrade pip
	uv pip sync pyproject.toml
	pip install -e .

run:
	python -m src.run