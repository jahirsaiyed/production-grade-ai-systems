.PHONY: setup ingest eval judge gate test

setup:
	pip install -r requirements.txt

ingest:
	python ingest.py

eval:
	python eval_harness.py

judge:
	python judge.py

gate:
	python gate.py

test:
	pytest --cov=app --cov-report=term-missing
