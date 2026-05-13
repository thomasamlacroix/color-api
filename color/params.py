"""
Global parameters
"""
import os


#Model Input size
IMAGE_SIZE = (256, 256)


MODEL_TARGET = os.environ.get("MODEL_TARGET")
MODEL_NAME = os.environ.get("MODEL_NAME").strip() if os.environ.get("MODEL_NAME") else ""
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH").strip() if os.environ.get("LOCAL_MODEL_PATH") else ""

TOKEN= os.environ.get("CONN_TOKEN").strip() if os.environ.get("CONN_TOKEN") else ""
