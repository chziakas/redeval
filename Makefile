#!make


.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: clean
clean:
	git clean -fxd

.PHONY: poetry-update
poetry-update:
	poetry update

.PHONY: install
poetry-install:
	poetry install --with dev --sync

.PHONY: export-requirements
poetry-export-requirements:
	poetry export --without-hashes --output=requirements.txt

.DEFAULT_GOAL := default

default: poetry-install
