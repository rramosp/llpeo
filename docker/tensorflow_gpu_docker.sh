#!/bin/sh
sudo docker run --rm --runtime=nvidia -it --gpus all -p 8811:8811 -v /home/rlx:/mnt -v /opt/data:/opt/data rlx/tf
