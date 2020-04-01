#!/bin/bash

docker run --net=host -it -u $(id -u):$(id -g) --rm \
 -v $(realpath ~/develop/Generative-Adversarial-Network-Tutorial):/tf/notebooks \
 --runtime=nvidia tensorflow/tensorflow:nightly-gpu-py3-jupyter


