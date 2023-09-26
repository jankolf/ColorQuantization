#!/bin/bash

for file in $1/*
do
    if [[ "$file" == *".yaml" ]]; then
        torchrun --standalone --nnodes=1 --nproc-per-node=4 ./train.py "$file"
    fi
done
