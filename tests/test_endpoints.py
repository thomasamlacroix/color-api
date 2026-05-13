"""
Test API endpoints
"""

import pytest
import base64
import io
from PIL import Image
from httpx import AsyncClient, ASGITransport


TEST_IMG = 'tests/images/image0001_bw.jpg'



@pytest.mark.asyncio
async def test_root_is_up():
    from color.api.color_api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_ok():
    from color.api.color_api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.json() == {"API": "OK"}


@pytest.mark.asyncio
async def test_predict_color_is_up():
    from color.api.color_api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_color_is_dict():
    from color.api.color_api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_predict_color_has_key():
    from color.api.color_api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
    assert response.json().get('img_bw_resized', False)
    assert response.json().get('img_reconstructed', False)


@pytest.mark.asyncio
async def test_predict_color_decoding():
    from color.api.color_api import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        files = {'file': open(TEST_IMG, 'rb')}
        response = await ac.post("/predict_color/", files=files)
        json_result = response.json()

        img_data = base64.b64decode(json_result['img_bw_resized'])
        img_bw_resized = Image.open(io.BytesIO(img_data))
        img_bw_resized.save('tests/images/img_bw_resized.png')

        img_data = base64.b64decode(json_result['img_reconstructed'])
        img_reconstructed = Image.open(io.BytesIO(img_data))
        img_reconstructed.save('tests/images/img_reconstructed.png')
