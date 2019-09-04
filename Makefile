# Black magic to get module directories
modules := $(foreach initpy, $(foreach dir, $(wildcard *), $(wildcard $(dir)/__init__.py)), $(realpath $(dir $(initpy))))

echo:
	@echo $(modules)

pytest:
	# Just testing. No code coverage.
	pipenv run pytest -v -l -m "not integration"
