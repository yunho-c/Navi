# FROM ubuntu:16.04
FROM ubuntu:18.04
# FROM ubuntu:20.04
LABEL maintainer caffe-maint@googlegroups.com

ENV TZ=US
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=1.0

# RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
#     pip install --upgrade pip && \
#     cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
#     mkdir build && cd build && \
#     cmake -DCPU_ONLY=1 .. && \
#     make -j"$(nproc)"


RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git .
RUN pip install --upgrade pip
RUN cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd ..
# add python 3
RUN apt update -y && apt install python3-pip -y
RUN pip3 install --upgrade pip
RUN cd python && for req in $(cat requirements.txt) pydot; do pip3 install $req; done && cd ..
RUN pip3 install scipy opencv-contrib-python

RUN mkdir build && cd build
RUN cmake -DCPU_ONLY=1
# ..
RUN make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
