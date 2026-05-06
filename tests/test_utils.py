"""
Testing image preprocessing
"""

# import pytest
from PIL import Image
from color.params import IMAGE_SIZE
from color.utils import resize_image

TEST_IMG = '/images/image0001.jpg'

def test_resize_image():
    with Image.open(TEST_IMG) as img:
        img = resize_image(img)
        height, width = img.size
        assert IMAGE_SIZE == (height, width)
