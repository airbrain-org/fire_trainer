#!/bin/bash

docker run --net=host -it -u $(id -u):$(id -g) --rm \
 -v $(realpath /media/jywilson/DATA/development):/tf/notebooks \
 --runtime=nvidia jywilson/tensorflow:nightly-gpu-py3-jupyter-keras


