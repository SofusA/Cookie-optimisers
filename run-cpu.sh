#! /bin/bash
sudo docker run -it -p 8888:8888 -v `pwd`:/work sofusa/pytorch-jupyter ./docker/jupyter_run.sh
