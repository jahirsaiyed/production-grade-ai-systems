.PHONY: setup train eval gate test

setup:
	pip install -r requirements.txt

train:
	python train_model.py

eval:
	python eval_harness.py

gate:
	python gate.py

test:
	pytest --cov=. --cov-report=term-missing --cov-config=.coveragerc --cov-fail-under=80
