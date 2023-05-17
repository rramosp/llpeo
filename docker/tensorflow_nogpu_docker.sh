#!/bin/sh
docker run --rm -it -p 8811:8811 -v /home/rlx:/mnt -v /home/rlx/data:/opt/data rlx/tf
