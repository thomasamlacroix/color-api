"""
Resize an image to a fixed (params.IMAGE_SIZE ) size
"""

from color.params import IMAGE_SIZE
from PIL import Image
import tensorflow as tf
import tensorflow_io as tfio

def resize_image(image):
    """
    Converts an image of size (N,M) to an image of size IMAGE_SIZE
    """
    # Read image
    # Resize image
    resized_image = image.resize(IMAGE_SIZE)
    return resized_image


def rgb_to_lab(image):
    image = tf.cast(image, tf.float32) / 255.0
    lab = tfio.experimental.color.rgb_to_lab(image)
    L = lab[:, :, :, :1] / 100.
    ab = lab[:, :, :, 1:] / 128.
    return (L, ab)
