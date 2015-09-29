#!/bin/sh

docker run \
  --rm \
  -p 8888:8888 \
  -v $PWD:/home/jovyan/notebooks/ \
  insighttoolkit/simpleitk-notebooks:2015-miccai
