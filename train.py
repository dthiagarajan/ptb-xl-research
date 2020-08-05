""" Trains a classification model on the PTB-XL dataset using PyTorch Lightning.
Example usage:
python train.py --model_name Simple1DCNN --fast_dev_run True
python train.py --model_name resnet18 --fast_dev_run True
To run with a single GPU:
python train.py --model_name resnet18 --fast_dev_run True --gpus 1
"""
import argparse
from pytorch_lightning import seed_everything, Trainer

from core.models import PTBXLClassificationModel


def get_args():
    """Argument parser running the SimCLR model.
    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Train a classification model on PTB-XL diagnostic superclass'
    )
    parser = PTBXLClassificationModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    seed_everything(31)
    model = PTBXLClassificationModel(**vars(args))
    trainer = Trainer.from_argparse_args(args, deterministic=True)
    trainer.fit(model)
