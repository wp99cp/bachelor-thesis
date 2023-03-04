FROM python:3.11.0-slim

ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Make dir '/projects/bachelor-thesis/data' and '/projects/bachelor-thesis/tmp'
RUN mkdir -p /projects/bachelor-thesis/data && mkdir -p /projects/bachelor-thesis/tmp/res

EXPOSE 5000

# Entrypoint
CMD gunicorn --bind :5000 --workers 1 --threads 2 --timeout 60 app:app