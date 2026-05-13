"""
Test API endpoints with container
"""

import pytest
import base64
import io
import os
from PIL import Image
from httpx import AsyncClient
import re
import subprocess

# TOKEN=os.environ.get("CONN_TOKEN").strip()

TEST_IMG = 'tests/images/image0001_bw.jpg'

TIMEOUT = 30

# Find the port the docker image is running on
image_name = f"{os.environ.get('DOCKER_IMAGE_NAME')}:dev"
# Use docker ps to list all running containers derived from $GAR_IMAGE:dev
docker_ps_command = f'docker ps --filter ancestor={image_name} --format "{{{{.Ports}}}}"'
docker_ps_output = subprocess.Popen(docker_ps_command,
                        shell=True,
                        stdout=subprocess.PIPE) \
                    .stdout.read().decode("utf-8")

# If we have an output, extract the port the container is running on
if docker_ps_output:
    docker_port = re.findall(":(\d{4})-", docker_ps_output)[0]
else:
    # If no output set docker_port to None
    # In the tests we'll assert docker_port exists
    docker_port = None
    # Print guidance for student that will show when running the test
    print("""
          \033[0;35m
          WARNING: We did not find a port with a docker container running

          Verify: - That your docker container is running
                  - The docker image was correctly named using $GAR_IMAGE:dev
                  - If your API is working locally, that it is running on a docker
                    container and not just using uvicorn locally
          \033[0m""")

# Assemble the service url
SERVICE_URL = f"http://localhost:{docker_port}"


@pytest.mark.asyncio
async def test_root_is_up():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_ok():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        response = await ac.get("/")
    assert response.json() == {"API": "OK"}


@pytest.mark.asyncio
async def test_predict_color_is_up():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_color_is_dict():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_predict_color_has_key():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
    assert response.json().get('img_bw_resized', False)
    assert response.json().get('img_reconstructed', False)


@pytest.mark.asyncio
async def test_predict_color_decoding():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
        json_result = response.json()

        img_data = base64.b64decode(json_result['img_bw_resized'])
        img_bw_resized = Image.open(io.BytesIO(img_data))
        img_bw_resized.save('tests/images/img_bw_resized.png')

        img_data = base64.b64decode(json_result['img_reconstructed'])
        img_reconstructed = Image.open(io.BytesIO(img_data))
        img_reconstructed.save('tests/images/img_reconstructed.png')
