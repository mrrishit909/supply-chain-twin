export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55103/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8103

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m sc.seed --if-empty
dev:            ## API with reload on :8103, jobs run by a local worker
	($(PY) -m core.jobs sc.api &) && $(PY) -m uvicorn sc.api:app --reload --port 8103
test:
	$(PY) -m pytest --cov=sc --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the six-step scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:
	$(PY) -m core.loadtest $(BASE) acme-viewer-demo 32 10 "GET /v1/map/state"
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test reset
