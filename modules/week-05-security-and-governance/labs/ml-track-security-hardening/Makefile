.PHONY: setup train model-card fairness-audit test run docker-build docker-run

setup:
	pip install -r requirements.txt

train:
	python train_model.py

model-card:
	python model_card.py

fairness-audit:
	python fairness_audit.py

test:
	pytest --cov=app --cov-report=term-missing --cov-fail-under=80 --cov-config=.coveragerc

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t ml-security-hardening-service .

docker-run:
	docker run --rm -p 8000:8000 -e JWT_SECRET_KEY -e MODEL_ENCRYPTION_KEY ml-security-hardening-service
