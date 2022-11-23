SHELL := /bin/bash
STAGED := $(shell git diff --cached --name-only --diff-filter=ACMR -- 'src/***.py' | sed 's| |\\ |g')

format:
	black .
	isort .

lint:
ifdef STAGED
	python -m pylint $(STAGED)
	python -m flake8 $(STAGED)
else
	@echo "No Staged Python File in the src folder"
endif

init:
	pip install -U pip
	pip install -e .
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

jupyter-kernel:
	python -m ipykernel install --user
