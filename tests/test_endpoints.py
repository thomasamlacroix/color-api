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

TEST_IMG = 'test/img/IMG_20240329_131211031.jpg'

HEALTH_EP = "/"
RELOAD_EP = "/reload/"
PREDICT_EP = "/predict_image/"
# TIMEOUT = 30



@pytest.mark.asyncio
async def test_root_is_up():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.json() == {"greeting": "Hello"}


@pytest.mark.asyncio
async def test_predict_is_up():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_is_dict():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_predict_has_key():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.json().get('fare', False)
