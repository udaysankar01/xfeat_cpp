#!/bin/bash
sudo docker run --net=host -e DISPLAY=$DISPLAY -v $(realpath .):/work -w /work -v $HOME/.Xauthority:/home/user/.Xauthority:rw --env="XDG_RUNTIME_DIR" -v ${XDG_RUNTIME_DIR}:${XDG_RUNTIME_DIR}:rw --rm --privileged --tty --volume /dev:/dev -it $1 /bin/bash
