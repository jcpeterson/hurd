MODULE_PATH=./hurd

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} + 
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

isort:
	sh -c "isort --skip-glob=.tox --recursive --balanced --use-parentheses --line-width=79 . "

lint:
	flake8 --exclude=.tox $(MODULE_PATH)

test: clean-pyc
	tox

build: clean-build
	python3 setup.py sdist
	python3 setup.py bdist_wheel --universal

release: build
	twine upload dist/*
