reinstall_package:
	@pip uninstall -y color_api || :
	@pip install -e .

run_api:
	uvicorn api.color_api:app --reload


run_interface:
	python interface/main.py


test_resize_image:
	pytest \
	test/api/test_utils.py::test_resize_image --asyncio-mode=strict -W "ignore"


test_root_is_up:
	pytest \
	test/api/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore"

test_root_returns_ok:
	pytest \
	test/api/test_endpoints.py::test_root_returns_ok --asyncio-mode=strict -W "ignore"

test_predict_color_is_up:
	pytest \
	test/api/test_endpoints.py::test_predict_color_is_up --asyncio-mode=strict -W "ignore"

test_predict_color_is_dict:
	pytest \
	test/api/test_endpoints.py::test_predict_color_is_dict --asyncio-mode=strict -W "ignore"

test_predict_color_has_key:
	pytest \
	test/api/test_endpoints.py::test_predict_color_has_key --asyncio-mode=strict -W "ignore"

test_predict_color_decoding:
	pytest \
	test/api/test_endpoints.py::test_predict_color_decoding --asyncio-mode=strict -W "ignore"


default: pylint pytest

pylint:
	find . -iname "*.py" -not -path "./test/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes
