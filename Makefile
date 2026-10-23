.PHONY: install seed api worker ui test lint eval smoke

PY ?= python

install:
	$(PY) -m pip install -e ".[ui,dev]"

seed:
	$(PY) -m lumen.seed

api:
	$(PY) -m uvicorn lumen.api.app:app --reload --port 8000

worker:
	$(PY) -m lumen.ingestion.worker

ui:
	streamlit run ui/streamlit_app.py

test:
	$(PY) -m pytest -q -rs

lint:
	ruff check . && ruff format --check .

eval:
	$(PY) -m eval.run_eval

smoke:
	$(PY) scripts/smoke.py http://localhost:8000
