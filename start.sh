#!/bin/bash

dir=$(dirname $(realpath -s $0))
dataset_root=$1
wandb docker-run -v $dir:/workspace -v $dataset_root:/datasets -it --ipc host  --shm-size 8G --gpus all sim2real /bin/bash
