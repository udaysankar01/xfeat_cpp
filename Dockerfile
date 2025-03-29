FROM ubuntu:20.04

ENV TX=Indian

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

RUN apt-get install -y git

WORKDIR /
RUN apt-get install -y libssl-dev cmake build-essential libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at
RUN git clone https://github.com/IntelRealSense/librealsense.git
WORKDIR /librealsense
RUN mkdir build
WORKDIR /librealsense/build
RUN cmake ..
RUN make -j11
RUN make install

RUN apt-get update

RUN apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update

WORKDIR /

RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt-get install -y python2.7-dev python3.6-dev python-dev python-numpy python3-numpy
RUN apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
RUN apt-get install -y libv4l-dev v4l-utils qv4l2 curl 
RUN apt-get install -y zip unzip

RUN curl -L https://github.com/opencv/opencv/archive/4.1.1.zip -o opencv-4.1.1.zip
RUN curl -L https://github.com/opencv/opencv_contrib/archive/4.1.1.zip -o opencv_contrib-4.1.1.zip
RUN unzip opencv-4.1.1.zip
RUN unzip opencv_contrib-4.1.1.zip
WORKDIR /opencv-4.1.1/

RUN sed -i 's/include <Eigen\/Core>/include <eigen3\/Eigen\/Core>/g' modules/core/include/opencv2/core/private.hpp

RUN mkdir build
WORKDIR /opencv-4.1.1/build
RUN cmake -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.1/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make -j11
RUN make install
RUN export PYTHONPATH=$PYTHONPATH:'$PWD'/python_loader/

RUN apt-get install -y wget
# Without GPU
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip
# With GPU

RUN mv libtorch /usr/lib/
RUN apt-get install -y libeigen3-dev

WORKDIR /work/