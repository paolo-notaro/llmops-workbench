HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: install test demo report case-studies api clean

install:
	poetry config virtualenvs.in-project false --local
	poetry config virtualenvs.path .venvs --local
	poetry install

test:
	poetry run pytest

demo:
	poetry run python scripts/run_demo.py

report:
	poetry run python scripts/build_report.py

case-studies:
	poetry run python scripts/build_case_studies.py

api:
	poetry run uvicorn llmops_workbench.app:app --host $(HOST) --port $(PORT)

clean:
	rm -rf reports .pytest_cache *.egg-info llmops_workbench/__pycache__ tests/__pycache__ scripts/__pycache__
