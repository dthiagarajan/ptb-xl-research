
# Run all unit tests
test:
	pytest -m "not functional" --cov-config=.coveragerc --cov-report=term --cov=.

# Run test in summary format. If p=<filename> is specified test only that file
test-summary:
	pytest -m "not functional" --tb=line -v $(p)

# Run all functional tests
test-functional:
	pytest -m "functional" --tb=line -v -s
