"""
Test API endpoints with Google Cloud
"""

import pytest
import base64
import io
import os
from PIL import Image
from httpx import AsyncClient
import re
import subprocess

TOKEN = os.environ.get("CONN_TOKEN").strip()

TEST_IMG = 'tests/images/image0001_bw.jpg'

TIMEOUT = 30

# Assemble the service url
SERVICE_URL = os.environ.get('SERVICE_URL')


if not SERVICE_URL:
    # Print guidance that will show when running the test
    print("""
          \033[0;35m
          WARNING: You did not set a SERVICE URL

          1. In your .env, set SERVICE_URL to the url of your Cloud Run endpoint
          2. Do a "direnv reload"
          3. Re-run the test
          \033[0m""")


@pytest.mark.asyncio
async def test_root_is_up():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_ok():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        response = await ac.get("/")
    assert response.json() == {"API": "OK"}


@pytest.mark.asyncio
async def test_predict_color_is_up():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        headers = {'token': TOKEN}
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files,
                                 headers=headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_color_is_dict():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        headers = {'token': TOKEN}
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files,
                                 headers=headers)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_predict_color_has_key():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        headers = {'token': TOKEN}
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files,
                                 headers=headers)
    assert response.json().get('img_bw_resized', False)
    assert response.json().get('img_reconstructed', False)


@pytest.mark.asyncio
async def test_predict_color_decoding():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=TIMEOUT) as ac:
        headers = {'token': TOKEN}
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files,
                                 headers=headers)
        json_result = response.json()

        img_data = base64.b64decode(json_result['img_bw_resized'])
        img_bw_resized = Image.open(io.BytesIO(img_data))
        img_bw_resized.save('tests/images/img_bw_resized.png')

        img_data = base64.b64decode(json_result['img_reconstructed'])
        img_reconstructed = Image.open(io.BytesIO(img_data))
        img_reconstructed.save('tests/images/img_reconstructed.png')
