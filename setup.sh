#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Download, extract, and install bore
wget https://github.com/ekzhang/bore/releases/download/v0.4.1/bore-v0.4.1-i686-unknown-linux-musl.tar.gz
tar -xf bore-v0.4.1-i686-unknown-linux-musl.tar.gz
rm -f bore-v0.4.1-i686-unknown-linux-musl.tar.gz
cp bore /usr/bin/bore
rm -rf bore

# Install Python 3.9 and related packages
apt-get install -y --no-install-recommends python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-distutils-extra
apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Create symlinks for python3 and python
ln -s /usr/bin/python3.9 /usr/local/bin/python3
ln -s /usr/bin/python3.9 /usr/local/bin/python

# Update packages and install pip
apt-get update -y
apt-get install -y python3-pip
python3.9 -m pip install --upgrade pip

# Install redis-server
sudo apt install lsb-release curl gpg
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg

sudo apt-get update
sudo apt-get install -y redis
sudo apt-get install -y redis-stack-server
sudo apt-get install -y  language-pack-id
LANGUAGE=en_US.UTF-8
LANG=en_US.UTF-8
LC_ALL=en_US.UTF-8
locale-gen en_US.UTF-8

# Start redis-server in the background
nohup redis-stack-server start > redis-debug.log &

# Install vllm package
python3.9 -m pip install langchain-cli pydantic==1.10.13
python3.9 -m pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7 jq
python3.9 -m pip install /home/mistral-rag/packages/rag-redis
# langchain app new mistral-rag --package rag-redis

# Start bore in the background
# nohup bore local 8002 --to bore.pub -p 32532 > tunnel_srv.txt &
# nohup bore local 6379 --to bore.pub -p 26123 > redis_tunnel_srv.txt &