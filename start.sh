#!/bin/bash

dir=$(dirname $(realpath -s $0))
docker run -v $dir:/workspace -v /mnt/sda/datasets:/datasets -it --gpus all sim2real /bin/bash
