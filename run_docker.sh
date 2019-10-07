#!/bin/bash

# -u $(id -u):$(id -g) \
# -u 0:0 \

# --runtime=nvidia jywilson/tensorflow:nightly-gpu-py3-jupyter-keras

#docker run --net=host -it --rm \
# -u $(id -u):$(id -g) \
# -v $(realpath ~/develop):/tf/notebooks \
# tensorflow/tensorflow

docker run --net=host -it --rm \
 -v $(realpath ~/develop):/home/jywilson/develop jywilson/tensorflow-airbrain:latest




