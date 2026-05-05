import pytest
from PIL import Image
from helper import params

TEST_IMG = 'test/img/IMG_20240329_131211031.jpg'

def test_resize_image():
    # This doesn't load the pixel data, just the header
    with Image.open(TEST_IMG) as img:
        width, height = img.size
        assert params.IMAGE_SIZE == (width, height)
