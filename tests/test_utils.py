"""
Testing image preprocessing
"""

import numpy as np
from PIL import Image
from color.params import IMAGE_SIZE
from color.utils import resize_image, rgb_to_lab

TEST_IMG = 'tests/images/image0001_bw.jpg'

def test_resize_image():
    with Image.open(TEST_IMG) as img:
        img = resize_image(img)
        height, width = img.size
        assert IMAGE_SIZE == (height, width)


def test_rgb_to_lab():
    with Image.open(TEST_IMG) as img:
        L, ab = rgb_to_lab(np.expand_dims(img, axis=0))
        assert L.shape[-1] == 1
        assert L.shape[1:-1] == img.size
        assert ab.shape[-1] == 2
        assert ab.shape[1:-1] == img.size
