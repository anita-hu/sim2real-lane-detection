FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         libopencv-dev \
         python-opencv \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         axel \
         zip \
         unzip
RUN pip install tensorboard
RUN pip install opencv-python

RUN useradd -m dev
USER dev
