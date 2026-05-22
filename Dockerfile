FROM python:3.12-slim

COPY color /color
#COPY /models/baseline.joblib /models/baseline.joblib
COPY /models/transfer_learning_model.joblib /models/transfer_learning_model.joblib
COPY setup.py /setup.py
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

CMD uvicorn color.api.color_api:app --host 0.0.0.0 --port $PORT
