.PHONY: setup train test mlflow-ui

setup:
	pip install -r requirements.txt

train:
	python train_with_tracking.py

test:
	pytest --cov=. --cov-report=term-missing --cov-config=.coveragerc

mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns
