# =============================================================================
# AI:OS — Makefile
# =============================================================================
#
# Common targets for development, testing, and deployment.
#
# Usage:
#   make help        Show available targets
#   make build       Build Docker images
#   make up          Start services
#   make test        Run test suite
#   make lint        Run linters
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help build up down restart logs shell test lint clean fmt check \
        status ps env-check deploy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMPOSE     := docker compose
IMAGE_NAME  := aios
IMAGE_TAG   := latest
PYTHON      := python3
PYTEST      := $(PYTHON) -m pytest
RUFF        := $(PYTHON) -m ruff

# Detect .env file
ENV_FILE := $(wildcard .env)
ifeq ($(ENV_FILE),)
  $(warning ⚠  No .env file found — copy .env.example to .env and configure)
endif

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help: ## Show this help message
	@echo ""
	@echo "  AI:OS — Agentic Intelligence Operating System"
	@echo "  =============================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
build: ## Build Docker image(s)
	$(COMPOSE) build

up: ## Start all services (detached)
	$(COMPOSE) up -d

up-full: ## Start all services including tool-launcher
	$(COMPOSE) --profile full up -d

down: ## Stop and remove containers
	$(COMPOSE) down

restart: down up ## Restart all services

logs: ## Tail logs from all services
	$(COMPOSE) logs -f --tail=100

logs-aios: ## Tail logs from aios service only
	$(COMPOSE) logs -f --tail=100 aios

shell: ## Open a shell in the aios container
	$(COMPOSE) exec aios /bin/bash || $(COMPOSE) run --rm aios /bin/bash

ps: status ## Alias for status
status: ## Show running containers and health
	$(COMPOSE) ps

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------
test: ## Run test suite
	PYTHONPATH=. $(PYTEST) tests/ -v --tb=short -x 2>/dev/null || \
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

test-ci: ## Run tests with JUnit output (CI-friendly)
	PYTHONPATH=. $(PYTEST) tests/ -v --tb=short --junitxml=test-results.xml

lint: ## Run linters (ruff)
	$(RUFF) check . --select=E,W,F,I --ignore=E501,F401,F403 || \
	$(PYTHON) -m py_compile config.py settings.py model.py apps.py diagnostics.py && echo "✓ Core modules compile OK"

fmt: ## Auto-format code with ruff
	$(RUFF) format . || echo "ruff not installed — install with: pip install ruff"

check: lint test ## Run linters then tests

# ---------------------------------------------------------------------------
# Local (non-Docker) run
# ---------------------------------------------------------------------------
run: ## Run AI:OS locally (boot mode)
	PYTHONPATH=. $(PYTHON) aios_cli boot

run-status: ## Run AI:OS status check locally
	PYTHONPATH=. $(PYTHON) aios_cli status

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env-check: ## Verify required environment variables are set
	@echo "Checking environment..."
	@test -f .env && echo "  ✓ .env file exists" || echo "  ✗ .env file missing (copy .env.example)"
	@$(PYTHON) -c "from settings import settings; \
		print('  ✓ Settings loaded'); \
		print(f'    Network calls: {settings.allow_httpx_network}'); \
		print(f'    USPTO configured: {settings.uspto_credentials() is not None}'); \
		print(f'    Stripe configured: {settings.stripe_configured()}')" 2>/dev/null \
		|| echo "  ⚠ Could not load settings (check imports)"

install: ## Install Python dependencies locally
	pip install -r requirements.txt
	pip install -r requirements-dev.txt 2>/dev/null || true
	@echo "✓ Dependencies installed"

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------
deploy: ## Run deployment script
	bash scripts/deploy.sh

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove build artifacts, caches, temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	find . -type f -name "*.html.tmp" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ .pytest_cache/ .ruff_cache/ test-results.xml
	@echo "✓ Cleaned"

clean-docker: ## Remove Docker containers, images, and volumes
	$(COMPOSE) down -v --rmi local
	@echo "✓ Docker artifacts removed"

clean-all: clean clean-docker ## Remove everything (Python + Docker)
