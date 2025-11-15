FROM python:3.11
WORKDIR /usr/src/web

RUN apt-get update && apt-get install netcat-traditional -y
RUN python -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r ./requirements.txt

ARG MEDIA_ROOT
ARG NN_MODEL_FOLDER
ARG BASE_MODEL_FOLDER
ARG BASE_MODEL_WEIGHTS
ENV MEDIA_ROOT=$MEDIA_ROOT
ENV NN_MODEL_FOLDER=$NN_MODEL_FOLDER
ENV BASE_MODEL_FOLDER=$BASE_MODEL_FOLDER
ENV BASE_MODEL_WEIGHTS=$BASE_MODEL_WEIGHTS
ENV BASE_MODEL_FOLDER_PATH=${MEDIA_ROOT}${NN_MODEL_FOLDER}${BASE_MODEL_FOLDER}
ENV BASE_MODEL_PATH=${BASE_MODEL_FOLDER_PATH}${BASE_MODEL_WEIGHTS}
RUN echo "${MEDIA_ROOT}"
RUN mkdir -p ${MEDIA_ROOT}
RUN echo "${BASE_MODEL_FOLDER_PATH}"
RUN mkdir -p ${BASE_MODEL_FOLDER_PATH}

# RUN mkdir /usr/src/web/static_files && mkdir /usr/src/web/static_files/map
RUN pip install coreapi pyyaml imagecodecs[all]
ENV PYTHONUNBUFFERED=1
COPY entrypoint.sh .
COPY medweb/ /usr/src/web/server/
ENTRYPOINT [ "./entrypoint.sh" ]
