#!/bin/bash

# -u $(id -u):$(id -g) \
# -u 0:0 \

# --runtime=nvidia jywilson/tensorflow:nightly-gpu-py3-jupyter-keras

#docker run --net=host -it --rm \
# -u $(id -u):$(id -g) \
# -v $(realpath ~/develop):/tf/notebooks \
# tensorflow/tensorflow

# -u $(id -u):$(id -g) \

docker run --net=host -it --rm --privileged \
 -v $(realpath ~/develop):/home/jywilson/develop -v /dev/bus/usb:/dev/bus/usb \
 jywilson/tensorflow-airbrain:latest




