# EPBA Automation Makefile

.PHONY: all build up down clean ingest logs

all: build up

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

clean:
	docker-compose down -v
	rm -rf logs/*
	rm -rf data/chroma_db/*
	find . -type d -name "__pycache__" -exec rm -rf {} +

ingest:
	@echo "Running Ingestion Process (Cleaning DB first)..."
	docker-compose exec vector_agent python services/vector_agent/src/ingest.py

logs:
	tail -f logs/**/*.json

test:
	@echo "Running tests..."
	docker-compose up -d
	# Add actual test command here, e.g. python tests/end_to_end.py
