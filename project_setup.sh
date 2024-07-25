#!/usr/bin/env bash

# install all required dependencies
sudo apt-get update
sudo apt install libopencv-dev

# setup libtorch
mkdir -p thirdparty
cd thirdparty
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule update --init --recursive
git submodule sync
python tools/build_libtorch.py
cd ../..