#!/bin/sh

IMAGE_NAME=insighttoolkit/simpleitk-notebooks:2015-miccai

# Sets up Docker on Debian 8 Jessie.
echo 'deb http://http.debian.net/debian jessie-backports main' >> /etc/apt/sources.list
apt-get update
apt-get install -y docker.io
docker run --rm hello-world

docker pull jupyter/minimal:latest
docker pull jupyter/tmpnb:latest
docker pull $IMAGE_NAME
export TOKEN=$( head -c 30 /dev/urandom | xxd -p )
docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN --name=proxy jupyter/configurable-http-proxy --default-target http://127.0.0.1:9999
docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN \
	-v /var/run/docker.sock:/docker.sock \
	--name tmpnb \
		jupyter/tmpnb \
			 python orchestrate.py \
				--container_port=8888 \
				--cpu_shares=2 \
				--max_dock_workers=40 \
				--mem_limit=2048m \
				--image=$IMAGE_NAME \
				--command="sh -c 'jupyter notebook --no-browser --NotebookApp.base_url={base_path} --ip=0.0.0.0 --port {port}'"
