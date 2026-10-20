.DEFAULT_GOAL := help


.PHONY: help
help: ## Show help for each target
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: venv
venv:  ## create or sync the uv environment
	uv sync --python 3.13 --extra dev --extra nb --extra cli

.PHONY: pre-commit
pre-commit:  ## install the pre-commit git hooks
	uv run --extra dev pre-commit install

.PHONY: tests
tests:  ## run unit tests (eg. with target: make tests target=test_notebook.py::test_func)
	uv run --extra dev --extra nb --extra cli pytest tests/$(target)
