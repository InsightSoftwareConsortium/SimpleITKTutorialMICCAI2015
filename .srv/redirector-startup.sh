#!/usr/bin/env bash
apt-get update

apt-get install -y \
  git \
  curl \
  python-tornado \
  vim
mkdir -p /srv
cd /srv/
git clone https://github.com/thewtex/tmpnb-redirector.git
cd tmpnb-redirector/
./run.sh
