.PHONY: setup ingest test run benchmark docker-build docker-run

setup:
	pip install -r requirements.txt

ingest:
	python ingest.py

test:
	pytest --cov=app --cov-report=term-missing

run:
	uvicorn app.main:app --reload

benchmark:
	python benchmark.py

docker-build:
	docker build -t semantic-cache-service .

docker-run:
	docker run --rm -p 8000:8000 -e OPENAI_API_KEY semantic-cache-service
