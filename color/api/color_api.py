"""
API endpoints
"""

import sys
import os
import io
import base64
from PIL import Image
from color.params import IMAGE_SIZE, TOKEN
from color.utils import resize_image, rgb_to_lab
from color.registry import load_model
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse


app = FastAPI()
app.model = None


@app.get("/")
def root():
    """
    API health check
    """
    if app.model is None:
        app.model =  load_model()
    return {
        'API': 'OK'
    }




@app.post("/predict_color/")
async def predict_color(file: UploadFile = File(...),
                  request: Request=None):
    """
    Upload a greyscale image file(acepted types: '.jpg', '.jpeg', '.png', '.gif', '.bmp')
        return :
            "img_bw_resized":  The original greyscale image, resized to 256x256,
            "img_reconstructed": the predicted colorized image, with size 256x256,
    """
    headers = request.headers
    token = headers.get("token")
    # If there is no token or it does not match --> Error
    if TOKEN != token:
        return JSONResponse(
                    status_code=400,
                    content={
                        "ERROR":  "Missing or wrong token."
                    }
                )

    if app.model is None:
        app.model =  load_model()
        if app.model is None:
            return JSONResponse(
                    status_code=500,
                    content={
                        "ERROR":  "Error loading model."
                    }
                )
    try:
        if file.content_type is not None:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
        else:
            # If content_type is None, check by file extension
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            file_extension = os.path.splitext(file.filename)[1].lower()

            # Validate file type
            if file_extension not in allowed_extensions:
                raise HTTPException(status_code=400, detail="File must be an image")

        contents = file.file
        img_bw = Image.open(contents)
        #Resized image
        img_bw_resized = resize_image(img_bw)
        #Convert into array
        img_bw_resized_array = np.array(img_bw_resized)


        shape = img_bw_resized_array.shape

        if shape[:-1] != IMAGE_SIZE:
            return JSONResponse(
                status_code=400,
                content={
                    "ERROR":  "The image size is not as expected.",
                    "expected": IMAGE_SIZE,
                    "received": shape[:-1],
                }
            )

        #RGB to Lab
        if shape[-1] == 1:
            #Repeat the grayscale array 3 times to mimic an RGB image
            fake_rgb = np.repeat(img_bw_resized_array, 3, axis=-1)  #(H, W, 3)
            L, _ = rgb_to_lab(np.expand_dims(fake_rgb, axis=0))
        elif shape[-1] == 3:
            L, _ = rgb_to_lab(np.expand_dims(img_bw_resized_array, axis=0))
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "ERROR":  "The expected number of channels is 1 or 3."
                }
            )

        ab_pred = app.model.predict(L)

        img_lab_reconstructed = tf.concat([L * 100., ab_pred * 128.], axis=-1)
        img_rgb_reconstructed = tfio.experimental.color.lab_to_rgb(img_lab_reconstructed)
        img_rgb_reconstructed = np.squeeze(img_rgb_reconstructed, axis=0)
        img_rgb_reconstructed_processed = Image.fromarray(np.uint8(img_rgb_reconstructed*255))

        # Encode image as base64
        #converts images to bytes
        #Original B&W image in PNG format
        img_byte_arr = io.BytesIO()
        img_bw_resized.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_bw_resized_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        #Reconstructed color image in PNG format
        img_byte_arr = io.BytesIO()
        img_rgb_reconstructed_processed.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_reconstructed_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        content = {
            "img_bw_resized": img_bw_resized_base64,
            "img_reconstructed": img_reconstructed_base64
        }

        return JSONResponse(status_code=200, content=content)

    except Exception as e:
        type, value, traceback = sys.exc_info()
        print(type, value, traceback )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()
