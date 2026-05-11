"""
Main to test predictions before implementing API endpoint
"""

from color.registry import load_model
from color.params import IMAGE_SIZE
from color.utils import resize_image, rgb_to_lab
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import io
import base64


def pred():
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict color image")

    img = resize_image(Image.open("tests/images/image0001.jpg"))
    img = np.array(img)

    L, ab = rgb_to_lab(np.expand_dims(img, axis=0))

    model = load_model()
    assert model is not None

    ab_pred = model.predict(L)

    img_lab_reconstructed = tf.concat([L * 100., ab_pred * 128.], axis=-1)

    img_rgb_reconstructed = tfio.experimental.color.lab_to_rgb(img_lab_reconstructed)

    #Convert into bytes and back to image
    img_rgb_reconstructed = np.squeeze(img_rgb_reconstructed, axis=0)
    # img_rgb_reconstructed_bytes = img_rgb_reconstructed.tobytes()
    # img_rgb_reconstructed_processed =  Image.frombytes('RGB',
    #                                      IMAGE_SIZE,
    #                                      img_rgb_reconstructed_bytes)
    # print(img_rgb_reconstructed)
    img_rgb_reconstructed_processed = Image.fromarray(np.uint8(img_rgb_reconstructed*255))

    #Reconstructed color image in PNG format
    img_byte_arr = io.BytesIO()
    img_rgb_reconstructed_processed.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_reconstructed_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    img_data = base64.b64decode(img_reconstructed_base64)
    img_reconstructed = Image.open(io.BytesIO(img_data))

    print("\n✅ prediction done, plotting images")

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    ax[0].imshow(np.squeeze(L, axis=0), cmap='grey')
    ax[0].axis('off')
    ax[0].set_title('Input')

    # ax[1].imshow(np.squeeze(img_rgb_reconstructed,axis=0))
    ax[1].imshow(img_reconstructed)
    ax[1].axis('off')
    ax[1].set_title('Predicted')

    ax[2].imshow(img)
    ax[2].axis('off')
    ax[2].set_title('Original')

    plt.show()

    return


if __name__ == '__main__':
    pred()
