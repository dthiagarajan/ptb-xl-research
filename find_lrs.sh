#!/bin/sh

# Finds all LRs using the LR find algorithm, and stores them in lr_find_archive.json for future use

python3 train.py --max_epochs 0 --find_lr True --gpus 1 --logger False --model_name Simple1DCNN &&
python3 train.py --max_epochs 0 --find_lr True --gpus 1 --logger False --model_name resnet18 &&
python3 train.py --max_epochs 0 --find_lr True --gpus 1 --logger False --model_name resnet34 &&
python3 train.py --max_epochs 0 --find_lr True --gpus 1 --logger False --model_name resnext50_32x4d &&
python3 train.py --max_epochs 0 --find_lr True --gpus 1 --logger False --model_name resnext101_32x8d &&
python3 train.py --max_epochs 0 --find_lr True --gpus 1 --logger False --model_name wide_resnet50_2 &&
python3 train.py --max_epochs 0 --find_lr True --gpus 1 --logger False --model_name wide_resnet101_2
