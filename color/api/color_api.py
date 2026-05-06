"""
API endpoints
"""

#import numpy as np
#import pandas as pd
import sys
import os
import io
import base64
from PIL import Image
from color import params
from color.utils import resize_image
from color.registry import load_model, get_response
import numpy as np
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




@app.post("/generate_heatmap/")
async def predict_heatmap(file: UploadFile = File(...),
                  request: Request=None):
    """
    Upload a greyscale image file(acepted types: '.jpg', '.jpeg', '.png', '.gif', '.bmp')
        return :
            "img_bw_resized":  The original greyscale image, resized to 256x256,
            "img_reconstructed": the predicted colorized image, with size 256x256,
            "original_resized": original color image, resized to 256x256
    """
    # headers = request.headers
    # token = headers.get("token")
    # # If there is no token or it does not match --> Error
    # if params.TOKEN != token:
    #     return JSONResponse(
    #                 status_code=400,
    #                 content={
    #                     "ERROR":  "Misisng or wrong token."
    #                 }
    #             )

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
        image = Image.open(contents)
        #image processed
        processed_image = resize_image(image)

        X_pred = img_to_array(processed_image)

        shape = X_pred.shape
        if shape != params.IMAGE_SIZE + (3,):
            return JSONResponse(
                status_code=400,
                content={
                    "ERROR":  "The image size is not as expected.",
                    "expected": params.IMAGE_SIZE + (3,),
                    "received": shape,
                }
            )
        shape = (-1,) + shape

        original_image = X_pred.astype('uint8')
        X_pred = X_pred.reshape(shape)
        y_pred = app.model.predict(X_pred)

        #heatmap
        explainer = GradCAM(app.model)
        # X, Y=[1== fake, 0==real]
        explanations = explainer(X_pred, [[1,0]])
        heatmap = explanations[0]

        heatmap = heatmap.numpy()
        #move the values to the positive side
        heatmap = heatmap - heatmap.min()
        heatmap = np.uint8(255 * heatmap / heatmap.max())

        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
        overlay = cv2.addWeighted(original_image, 0.5, colored_heatmap, 0.3, 0)

        img_array = overlay.tobytes()
        img_processed =  Image.frombytes('RGB', params.IMAGE_SIZE, img_array)

        # Encode image as base64
        #converts images to bytes
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_proc_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        #heatmap in PNG format
        img_byte_arr = io.BytesIO()
        img_processed.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        heatmap_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        content = get_response(y_pred)

        content["image_resized"] = img_proc_base64
        content["heatmap"] = heatmap_base64
        return JSONResponse(
            status_code=200,
            content=content
        )

    except Exception as e:
        type, value, traceback = sys.exc_info()
        print(type, value, traceback )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()
