.PHONY: install install-dev install-gpu test test-cov lint run-api run-frontend \
        migrate docker-up docker-down docker-build clean eval

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

install-gpu: install
	pip install -r requirements-local-inference.txt

test:
	pytest

test-cov:
	pytest --cov=app --cov-report=term-missing

migrate:
	alembic upgrade head

migration:
	alembic revision --autogenerate -m "$(name)"

run-api:
	uvicorn app.main:app --reload --port 8000

run-frontend:
	streamlit run frontend/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

docker-down:
	docker compose down

eval:
	python -c "from evaluation.dataset import build_synthetic_dataset, save_dataset_to_disk; \
	           samples = build_synthetic_dataset(); save_dataset_to_disk(samples); \
	           print(f'Generated and saved {len(samples)} evaluation samples to data/samples/')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
