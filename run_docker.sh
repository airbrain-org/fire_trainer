#!/bin/bash

# -u $(id -u):$(id -g) \

docker run --net=host -it --rm \
 -v $(realpath /media/jywilson/DATA/development):/tf/notebooks \
 --runtime=nvidia jywilson/tensorflow:nightly-gpu-py3-jupyter-keras


