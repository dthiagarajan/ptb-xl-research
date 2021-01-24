""" Trains an unsupervised model on the PTB-XL dataset using PyTorch Lightning.
Example usage:
python train_unsupervised.py --model_name Simple1DCNN --fast_dev_run True
python train_unsupervised.py --model_name resnet18 --fast_dev_run True

To run with a single GPU:
python train_unsupervised.py --model_name resnet18 --fast_dev_run True --gpus 1
"""
import argparse
import os
import pandas as pd
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.distributed as dist

from core.callbacks import (
    BatchGradientVerificationCallback,
    BatchNormVerificationCallback,
    ModuleDataMonitor,
    ProgressBar
)
from core.data import PTBXLDataModule
from core.loggers import TensorBoardLogger, WandbLogger
from core.models import PTBXLUnsupervisedModel


def get_args():
    """Argument parser for training a PTB-XL classification model.

    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Train a classification model on PTB-XL diagnostic superclass'
    )
    parser.add_argument('--log_dir', type=str, default='./lightning_logs')
    parser.add_argument('--checkpoint_models', type=bool, default=False)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--find_lr', type=bool, default=False)
    parser = PTBXLUnsupervisedModel.add_model_specific_args(parser)
    parser = PTBXLDataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    seed_everything(31)

    data_module = PTBXLDataModule(
        args.data_dir,
        args.sampling_rate,
        args.task_name,
        unsupervised=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = PTBXLUnsupervisedModel(**vars(args))
    model.suppress_logging = False

    if args.find_lr:
        assert args.distributed_backend is None, 'LR find will not work properly distributed'
        try:
            lr_find_config = pd.read_json('./lr_find_archive.json', orient='index')
        except ValueError:
            print('Archive file does not exist, creating in this run.')
            lr_find_config = pd.DataFrame()
        if args.model_name not in lr_find_config.index:
            # Need to prepare dataloaders for LR find
            data_module.setup()
            print(f'Finding LR - note that specified LR ({args.lr}) is being overriden.')
            trainer = Trainer()
            # This will take longer than a single epoch would take, but worth it for a more
            # fine-grained LR check
            lr_finder = trainer.lr_find(
                model, train_dataloader=data_module.train_dataloader(), num_training=1000
            )
            suggested_lr = lr_finder.suggestion()
            print(f'Found best LR of {suggested_lr:0.5f}.')
            for key, item in vars(args).items():
                lr_find_config.loc[args.model_name, key] = item
            lr_find_config.loc[args.model_name, 'lr'] = suggested_lr
            lr_find_config.to_json('./lr_find_archive.json', orient='index')
        else:
            suggested_lr = float(lr_find_config.loc[args.model_name, 'lr'])
            print(f'Reading LR {suggested_lr} from archive config.')
        model.lr = suggested_lr
        # Need to manually update, similar to doc.
        # Reference: https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html
        model.hparams.lr = suggested_lr

    if args.logger_platform == 'wandb':
        logger = WandbLogger(project="ptb-xl")
    elif args.logger_platform == 'tensorboard':
        logger = TensorBoardLogger(args.log_dir, name='')
        model.log_dir = args.log_dir

    early_stopping_callback = EarlyStopping(
        verbose=True,
        monitor='val_epoch_loss',
        mode='min',
        patience=5
    ) if args.early_stopping else None
    checkpoint_callback = args.checkpoint_models and int(os.environ.get('LOCAL_RANK', 0)) == 0
    progress_bar_callback = ProgressBar()
    batch_gradient_verification_callback = BatchGradientVerificationCallback(
        output_mapping=lambda output: output['loss']
    )
    batch_norm_verification_callback = BatchNormVerificationCallback()
    module_monitor_callback = ModuleDataMonitor(submodules=True)

    # Resetting trainer due to some issue with threading otherwise
    trainer = Trainer.from_argparse_args(
        args, checkpoint_callback=checkpoint_callback, deterministic=True, logger=logger,
        callbacks=list(filter(
            lambda callback: callback is not None,
            [
                early_stopping_callback,
                progress_bar_callback,
                module_monitor_callback,
                batch_gradient_verification_callback,
                batch_norm_verification_callback
            ]
        ))
    )
    trainer.fit(model, data_module)

    if args.checkpoint_models and int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(f'Best model path: {checkpoint_callback.best_model_path}.')
