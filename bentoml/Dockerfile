FROM python:3.8.2-slim-buster
RUN pip install --upgrade pip
RUN apt-get update && \
    apt-get install -y gcc build-essential git && \
    apt-get clean 
RUN pip install bentoml boto3 pandas sklearn