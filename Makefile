.PHONY: test

test:
	python -m unittest -v tests/test_*.py

pylint:
	pylint --rcfile=./linting_config/pylint-configuration.pylintrc $(shell find ./punctfix/ -name "*.py" | xargs)
