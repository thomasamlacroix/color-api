"""
Resize an image to a fixed (params.IMAGE_SIZE ) size
"""

from color.params import IMAGE_SIZE
from PIL import Image

def resize_image(image):
    """
    Converts an image of size (N,M) to an image of size IMAGE_SIZE
    """
    # Read image
    # Resize image
    resized_image = image.resize(IMAGE_SIZE)
    return resized_image
