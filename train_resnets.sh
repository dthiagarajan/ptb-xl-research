#!/bin/sh

# python3 train.py --model_name resnet18 --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name resnet34 --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name resnet50 --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name resnet101 --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name resnet152 --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name resnext50_32x4d --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name resnext101_32x8d --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet18_2 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet34_2 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name wide_resnet50_2 --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
# python3 train.py --model_name wide_resnet101_2 --max_epochs 25 --distributed_backend ddp --mixup True --mixup_layer 2 --gpus 4 --checkpoint_models True &&
# # nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet18_4 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet34_4 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet50_4 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet101_4 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet18_8 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet34_8 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet50_8 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python3 train.py --model_name wide_resnet101_8 --max_epochs 25 --distributed_backend ddp --mixup True --gpus 4 --checkpoint_models True &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9
