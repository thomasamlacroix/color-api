FROM python:3.12-slim

COPY color /color
COPY /models/baseline.joblib /models/baseline.joblib
COPY setup.py  /setup.py
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

CMD uvicorn api.dfake_api:app --host 0.0.0.0
