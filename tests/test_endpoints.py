"""
Test API endpoints
"""

import pytest
import base64
import io
import os
from PIL import Image
from httpx import AsyncClient

# SERVICE_URL = os.environ.get("SERVICE_URL")
# TOKEN=os.environ.get("CONN_TOKEN").strip()

TEST_IMG = '/images/image0001.jpg'

# TIMEOUT = 30



@pytest.mark.asyncio
async def test_root_is_up():
    from color.api.color_api import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_ok():
    from color.api.color_api import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.json() == {"API": "OK"}


@pytest.mark.asyncio
async def test_predict_color_is_up():
    from color.api.color_api import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict_color/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_color_is_dict():
    from color.api.color_api import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict_color/")
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_predict_color_has_key():
    from color.api.color_api import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict_color/")
    assert response.json().get('colored_img', False)


@pytest.mark.asyncio
async def test_predict_color_decoding():
    from color.api.color_api import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict_color/")
        json_result = response.json()

        img_data = base64.b64decode(json_result['img_bw_resized'])
        img_bw_resized = Image.open(io.BytesIO(img_data))
        img_bw_resized.save('img_bw_resized.png')

        img_data = base64.b64decode(json_result['img_reconstructed'])
        img_reconstructed = Image.open(io.BytesIO(img_data))
        img_reconstructed.save('img_reconstructed.png')

        img_data = base64.b64decode(json_result['original_resized'])
        original_resized = Image.open(io.BytesIO(img_data))
        original_resized.save('original_resized.png')
