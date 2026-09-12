.PHONY: setup ingest test run docker-build docker-run

setup:
	pip install -r requirements.txt

ingest:
	python ingest.py

test:
	pytest --cov=app --cov-report=term-missing

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t rag-service .

docker-run:
	docker run --rm -p 8000:8000 -e OPENAI_API_KEY rag-service
