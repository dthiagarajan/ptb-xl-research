""" Trains a classification model on the PTB-XL dataset using PyTorch Lightning.
Example usage:
python train.py --model_name Simple1DCNN --fast_dev_run True
python train.py --model_name resnet18 --fast_dev_run True

To run with a single GPU:
python train.py --model_name resnet18 --fast_dev_run True --gpus 1

To run with a custom loss function:
python train.py --model_name Simple1DCNN --fast_dev_run True \
    --loss_function '2*focal_loss + f1_loss'
"""
import argparse
import os
import pandas as pd
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.distributed import find_free_network_port
import torch.distributed as dist

from core.callbacks import MultiMetricModelCheckpoint
from core.data import PTBXLDataModule
from core.loggers import TensorBoardLogger, WandbLogger
from core.models import PTBXLClassificationModel


def get_args():
    """Argument parser for training a PTB-XL classification model.

    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Test a classification model on PTB-XL diagnostic superclass'
    )
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser = PTBXLClassificationModel.add_model_specific_args(parser)
    parser = PTBXLDataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    seed_everything(31)
    print(f'Loading model from checkpoint...')
    model = PTBXLClassificationModel.load_from_checkpoint(args.model_checkpoint)
    model.model_checkpoint = args.model_checkpoint
    model.show_heatmaps = False
    print(f'...done.')
    data_module = PTBXLDataModule(
        args.data_dir, args.sampling_rate, args.task_name,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    data_module.setup()

    if args.logger_platform == 'wandb':
        logger = WandbLogger(project="ptb-xl")
    elif args.logger_platform == 'tensorboard':
        logger = TensorBoardLogger('./testing_logs', name='')
        model.suppress_logging = True
        model.log_dir = './testing_logs'

    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=logger)

    model.labels = data_module.labels  # Needed for confusion matrix labels
    trainer.test(model=model, datamodule=data_module)
