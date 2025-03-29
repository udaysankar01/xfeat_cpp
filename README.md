# C++ Implementation of XFeat

> This repo contains the C++ implementation of the CVPR 2024 paper [XFeat: Accelerated Features for Lightweight Image Matching](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html).

Original Repo: https://github.com/verlab/accelerated_features

Paper: https://arxiv.org/abs/2404.19174

![Image Matches](doc/image_matches.png)

## Prerequisite

In this project, the following packages are used. Make sure the right versions of libraries are installed and linked.

1. Tested in Ubuntu 22.04 / 20.04
2. Nvidia-driver-535 and CUDA Toolkit 12.2
3. gcc and g++ compilers 11.4.0
4. CMake 3.22.1 / 3.5
5. OpenCV 4.5.4 / 4.1.1
6. [libtorch](https://github.com/pytorch/pytorch): Please avoid using the pre-built version of libtorch since it will cause linking issues ([CXX11 ABI issue](https://github.com/pytorch/pytorch/issues/13541))

## Setup

To download the project:

```
git clone https://github.com/udaysankar01/xfeat_cpp
cd xfeat_cpp
```

To install the necessary packages (OpenCV and libtorch):

```bash
chmod +x project_setup.sh
./project_setup.sh
```

To build the project:

```bash
mkdir -p build
cd build
cmake ..
make -j4
```
To build the project using Docker
```bash 
sudo docker build -t xfeat . 
            or
./build_docker.sh
xhost + (in different terminal)
./run_docker.sh xfeat
mkdir build 
cd build
cmake ..
make -j4
```

## Running

To perform matching between two images, use this command:

```bash
./build/examples/example /absolute/path/to/image1 /absolute/path/to/image2
```

Matching Example:

```bash
./build/examples/match $(pwd)/ref.png $(pwd)/tgt.png
```

Realtime Matching Example:

```bash
./build/examples/realtime_demo
```

Realsense Matching Example:

```bash
./build/realsense_demo
```

## Bibtex Citation

```
@misc{potje2024xfeatacceleratedfeatureslightweight,
      title={XFeat: Accelerated Features for Lightweight Image Matching},
      author={Guilherme Potje and Felipe Cadar and Andre Araujo and Renato Martins and Erickson R. Nascimento},
      year={2024},
      eprint={2404.19174},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.19174},
}
```
