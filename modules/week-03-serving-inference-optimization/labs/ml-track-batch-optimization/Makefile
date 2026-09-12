.PHONY: setup train test run docker-build docker-run benchmark

setup:
	pip install -r requirements.txt

train:
	python train_model.py

test:
	pytest --cov=app --cov-report=term-missing

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t fraud-detection-service .

docker-run:
	docker run --rm -p 8000:8000 fraud-detection-service

benchmark:
	python benchmark.py
