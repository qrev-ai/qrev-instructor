
test::
	pytest tests

format::
	toml-sort pyproject.toml

build:: test format
	poetry build

publish:: build
	poetry publish