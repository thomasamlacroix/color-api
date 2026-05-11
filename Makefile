reinstall_package:
	@pip uninstall -y color_rise_api || :
	@pip install -e .

run_api:
	uvicorn color.api.color_api:app --reload


run_interface:
	python interface/main.py


test_resize_image:
	pytest \
	tests/test_utils.py::test_resize_image --asyncio-mode=strict -W "ignore"


test_api_root:
	pytest \
	tests/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/test_endpoints.py::test_root_returns_ok --asyncio-mode=strict -W "ignore"


test_api_predict_color:
	pytest \
	tests/test_endpoints.py::test_predict_color_is_up --asyncio-mode=strict -W "ignore" \
	tests/test_endpoints.py::test_predict_color_is_dict --asyncio-mode=strict -W "ignore" \
	tests/test_endpoints.py::test_predict_color_has_key --asyncio-mode=strict -W "ignore" \
	tests/test_endpoints.py::test_predict_color_decoding --asyncio-mode=strict -W "ignore"

# test_api_on_docker:
# 	pytest \
# 	tests/api/test_docker_endpoints.py --asyncio-mode=strict -W "ignore"

# test_api_on_prod:
# 	pytest \
# 	tests/api/test_cloud_endpoints.py --asyncio-mode=strict -W "ignore"



default: pylint pytest

pylint:
	find . -iname "*.py" -not -path "./test/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes
