install:
	conda env create -f environment.yml

update-env:
	conda env update --name dvc-demo --file environment.yml --prune

test:
	python -m pytest -vv --cov=main --cov=mylib test_*.py

format:	
	black src/*/*.py

lint:
	pylint --disable=R,C src/*/*.py
	
container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	#deploy goes here
		
all: install lint test format deploy