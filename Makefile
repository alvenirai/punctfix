.PHONY: test

test:
	python -m unittest discover -p 'test_*.py' -s './tests/'
