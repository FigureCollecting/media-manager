SHELL := /bin/bash

export PYTHONWARNINGS=ignore

.PHONY: dev-up dev-down db-upgrade logs stubs test token

dev-up:
	docker compose up -d --build

dev-down:
	docker compose down -v

db-upgrade:
	docker compose exec api alembic upgrade head

logs:
	docker compose logs -f --tail=200 api worker

stubs:
	poetry run python scripts/gen_spine_stubs.py

test: stubs
	poetry run pytest -q

token:
	@[ -n "$(USER_ID)" ] || (echo "USER_ID is required"; exit 1)
	@curl -s -X POST http://localhost:8000/auth/dev-token -H 'Content-Type: application/json' \
		-d '{"user_id":"$(USER_ID)", "tenant_id":"$(TENANT_ID)"}' | jq -r .token

