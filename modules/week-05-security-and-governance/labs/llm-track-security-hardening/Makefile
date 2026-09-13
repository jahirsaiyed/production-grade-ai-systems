.PHONY: setup ingest test run docker-build docker-run

setup:
	pip install -r requirements.txt

ingest:
	python ingest.py

test:
	pytest --cov=app --cov-report=term-missing --cov-fail-under=80 --cov-config=.coveragerc

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t llm-security-hardening-service .

docker-run:
	docker run --rm -p 8000:8000 -e OPENAI_API_KEY -e JWT_SECRET_KEY llm-security-hardening-service
