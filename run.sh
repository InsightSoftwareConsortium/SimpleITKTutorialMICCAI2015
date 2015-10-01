#!/bin/sh

_MACHINE_NAME=$(docker-machine active 2> /dev/null || echo "default")

# if we need to use docker machine let us start the "default"
if ! docker info &> /dev/null; then
    command -v docker-machine &> /dev/null || (echo "Docker is not configured correctly. No host or docker-machine." && exit 1)
    if [ "$(docker-machine status $_MACHINE_NAME 2> /dev/null)" != "Running" ]; then
        echo "Starting machine $_MACHINE_NAME..."
        docker-machine start $_MACHINE_NAME || exit 1
    fi
    eval $(docker-machine env $_MACHINE_NAME --shell=sh)
fi

_MACHINE_IP=$(docker-machine ip ${_MACHINE_NAME} 2> /dev/null || echo "localhost" )
_URL="http://${_MACHINE_IP}:8888"

echo "\n\nSetting up Docker Jupyter Notebook at ${_URL}\n\n"


docker run \
  --rm \
  -p 8888:8888 \
  -v $PWD:/home/jovyan/notebooks/ \
  insighttoolkit/simpleitk-notebooks:2015-miccai

# vim: noexpandtab shiftwidth=4 tabstop=4 softtabstop=0
