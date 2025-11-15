FROM python:3.11
WORKDIR /usr/src/web/dj_nnapi

RUN apt-get update && apt-get install netcat-traditional -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip

# RUN rm -rf /opt/conda/lib/python3.8/site-packages/ruamel*

COPY ./requirements.txt .
RUN pip install -r ./requirements.txt

ARG MEDIA_ROOT
ENV MEDIA_ROOT=$MEDIA_ROOT

ENV PYTHONUNBUFFERED=1
RUN pip install imagecodecs[all]
COPY entrypoint.sh entrypoint.sh
COPY dj_nnapi/ dj_nnapi/
ENTRYPOINT [ "./entrypoint.sh" ]

