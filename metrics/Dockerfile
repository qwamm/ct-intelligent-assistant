FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
WORKDIR /usr/src/web/metrics
RUN python -m pip install --upgrade pip
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt
# ENV PYTHONUNBUFFERED=1