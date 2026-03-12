# Makefile for inertialai-chroma


# Constants
# -----------------------------------------------

# Variables
UV := uv
PYTHON := $(UV) run python
PYTEST := $(UV) run pytest
RUFF := $(UV) run ruff
MYPY := $(UV) run mypy
PREK := $(UV) run prek
BUILD := $(UV) run build

# Directories
SRC_DIR := src/inertialai_chroma
TESTS_DIR := tests

# Package metadata extracted from pyproject.toml
PACKAGE_NAME := `sed -n 's/^ *name.*=.*"\([^"]*\)".*/\1/p' pyproject.toml`
PACKAGE_VERSION := `sed -n 's/^ *version.*=.*"\([^"]*\)".*/\1/p' pyproject.toml`


# Introspection Targets
# -----------------------------------------------

.PHONY: help
help: header targets

.PHONY: header
header:
	@echo "\033[34mEnvironment\033[0m"
	@echo "\033[34m---------------------------------------------------------------\033[0m"
	@printf "\033[33m%-23s\033[0m" "PACKAGE_NAME"
	@printf "\033[35m%s\033[0m" $(PACKAGE_NAME)
	@echo ""
	@printf "\033[33m%-23s\033[0m" "PACKAGE_VERSION"
	@printf "\033[35m%s\033[0m" $(PACKAGE_VERSION)
	@echo "\n"

.PHONY: targets
targets:
	@echo "\033[34mDevelopment Targets\033[0m"
	@echo "\033[34m---------------------------------------------------------------\033[0m"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'


# Install Targets
# -----------------------------------------------

.PHONY: install
install: ## Install all dependencies
	$(UV) sync --all-groups --all-extras
	@make install-pre-commit

.PHONY: install-prod
install-prod: ## Install production dependencies
	$(UV) sync --no-dev

.PHONY: install-dev
install-dev: ## Install development dependencies
	$(UV) sync --only-dev


# Test Targets
# ----------------------------------------

.PHONY: test
test: ## Run unit tests with coverage
	$(PYTEST) $(TESTS_DIR)

.PHONY: test-dev
test-dev: ## Run unit tests with coverage and additional verbosity for debugging
	$(PYTEST) $(TESTS_DIR) -v

.PHONY: test-integration
test-integration: ## Run integration tests
	$(PYTEST) $(TESTS_DIR)/integration


# Checking, Linting, and Formatting Targets
# ----------------------------------------

.PHONY: type-check
type-check: ## Run type checker
	$(MYPY) $(SRC_DIR)

.PHONY: lint-format
lint-format: lint-fix sort-imports ## Run linter and formatter

.PHONY: lint
lint: ## Run linter
	$(RUFF) check

.PHONY: lint-fix
lint-fix: ## Run linter and fix any minor issues
	$(RUFF) check --fix

.PHONY: sort-imports
sort-imports: ## Sort imports
	$(RUFF) check --select I --fix
	make format

.PHONY: format
format: ## Run code formatter
	$(RUFF) format

.PHONY: check-lockfile
check-lockfile: ## Compares lock file with pyproject.toml
	$(UV) lock --check

.PHONY: update-lockfile
update-lockfile: ## Updates the lock file
	$(UV) lock

.PHONY: install-pre-commit
install-pre-commit: ## Installs the pre-commit hooks
	$(PREK) install --hook-type pre-commit --hook-type pre-push

.PHONY: uninstall-pre-commit
uninstall-pre-commit: ## Uninstalls the pre-commit hooks
	$(PREK) uninstall --hook-type pre-commit --hook-type pre-push

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	$(PREK)

.PHONY: check
check: lint type-check test ## Run all checks (lint, type-check, test)


# Build and Publish Targets
# ----------------------------------------

.PHONY: build
build: clean ## Build source and wheel distributions
	$(BUILD)

.PHONY: publish-test
publish-test: build ## Publish package to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

.PHONY: publish
publish: build ## Publish package to PyPI
	$(PYTHON) -m twine upload dist/*


# Clean Targets
# ----------------------------------------

.PHONY: clean
clean: ## Remove build artifacts and temporary files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .ruff_cache

.PHONY: nuke
nuke: ## Remove build artifacts, temporary files, the .venv directory and the uv.lock file
	make clean
	rm -rf .venv && rm -f uv.lock


# Example Setup Targets
# ----------------------------------------

EXAMPLES_DIR := examples

.PHONY: example-data-fetch
example-data-fetch: ## Download sample data for the example setup
	cd $(EXAMPLES_DIR) && ./data.sh fetch

.PHONY: example-data-fetch-force
example-data-fetch-force: ## Force re-download sample data for the example setup
	cd $(EXAMPLES_DIR) && ./data.sh fetch --force

.PHONY: example-data-clean
example-data-clean: ## Remove sample data from the example setup
	cd $(EXAMPLES_DIR) && ./data.sh clean

.PHONY: example-init-env
example-init-env: ## Copy .env.example to .env in the example setup directory (skips if .env already exists)
	@test -f $(EXAMPLES_DIR)/.env && echo "$(EXAMPLES_DIR)/.env already exists — skipping." || cp $(EXAMPLES_DIR)/.env.example $(EXAMPLES_DIR)/.env && echo "Created $(EXAMPLES_DIR)/.env — add your INERTIALAI_API_KEY."

.PHONY: example-up
example-up: ## Start the example setup (Chroma + demo app) in detached mode
	docker compose -f $(EXAMPLES_DIR)/docker-compose.yml up -d --build

.PHONY: example-down
example-down: ## Stop the example setup
	docker compose -f $(EXAMPLES_DIR)/docker-compose.yml down

.PHONY: example-logs
example-logs: ## Stream logs from the example setup
	docker compose -f $(EXAMPLES_DIR)/docker-compose.yml logs -f

.PHONY: example-reset
example-reset: example-down ## Stop the example setup and remove all Docker volumes (clears Chroma data)
	docker compose -f $(EXAMPLES_DIR)/docker-compose.yml down -v

.PHONY: example-clean
example-clean: example-reset example-data-clean ## Full cleanup: stop services, remove volumes, delete sample data
