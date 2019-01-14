rem docker run -p 8888:8888 -it --rm -v /d/development:/tf/notebooks tensorflow/tensorflow:nightly-gpu-py3-jupyter bash

docker run -p 8888:8888 -it --rm -v /d/development:/tf/notebooks jywilson/tensorflow:nightly-gpu-py3-jupyter-keras jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root



