FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
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
         unzip \
         g++
RUN pip install opencv-python
RUN pip install wandb

# Install OpenCV C++ for evaluation
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip && unzip opencv.zip
RUN mkdir -p build
RUN cd build && cmake  ../opencv-master && cmake --build .
RUN ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2

# Other evaluation deps
RUN pip install scikit-learn matplotlib

RUN useradd -m dev
USER dev
