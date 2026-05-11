""""
Module that contains the load_model and predict functions
"""

from color.params import MODEL_TARGET, LOCAL_MODEL_PATH, MODEL_NAME
import os
from colorama import Fore, Style
import numpy as np
import tensorflow as tf
from joblib import load



def load_model():
    """
    Return a saved model

    Return None (but do not Raise) if no model is found
    """
    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk

        local_model = MODEL_NAME
        model_path = LOCAL_MODEL_PATH
        local_model = os.path.join(model_path, local_model)

        if not local_model:
            return None


        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = load(local_model)

        print("✅ Model loaded from local disk")

        return latest_model




# def get_response(y_predict: np.ndarray):
#     """
#     Returns a dictionary with the images
#     """
#     predict_value = y_predict.item(0)
#     if predict_value > TRIGGER_VALUE:
#         fake_real = RESULTS[1]
#     else:
#         fake_real = RESULTS[0]

#     content={
#                 "fake_real":  fake_real,
#                 "predict_value": predict_value,
#             }
#     return content
