""" Trains a classification model on the PTB-XL dataset using PyTorch Lightning.
Example usage:
python train.py --model_name Simple1DCNN --fast_dev_run True
python train.py --model_name resnet18 --fast_dev_run True
To run with a single GPU:
python train.py --model_name resnet18 --fast_dev_run True --gpus 1
"""
import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from core.callbacks import MultiMetricModelCheckpoint
from core.models import PTBXLClassificationModel


def get_args():
    """Argument parser running the SimCLR model.
    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Train a classification model on PTB-XL diagnostic superclass'
    )
    parser.add_argument('--checkpoint_models', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--find_lr', action='store_true')
    parser = PTBXLClassificationModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    seed_everything(31)
    model = PTBXLClassificationModel(**vars(args))
    early_stopping_callback = EarlyStopping(
        verbose=True,
        monitor='val_epoch_loss',
        mode='min',
        patience=5
    ) if args.early_stopping else None
    checkpoint_callback = MultiMetricModelCheckpoint(
        verbose=True,
        monitors=['val_epoch_loss', 'val_epoch_auc', 'val_epoch_f_max'],
        modes=['min', 'max', 'max']
    ) if args.checkpoint_models else None

    if args.find_lr:
        trainer = Trainer()
        lr_finder = trainer.lr_find(model)
        suggested_lr = lr_finder.suggestion()
        print(f'Found best LR of {suggested_lr:0.5f}.')
        model.lr = suggested_lr
        # Need to manually update, similar to doc.
        # Reference: https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html
        model.hparams.lr = suggested_lr

    # Resetting trainer due to some issue with threading otherwise
    trainer = Trainer.from_argparse_args(
        args, checkpoint_callback=checkpoint_callback, early_stop_callback=early_stopping_callback,
        deterministic=True
    )
    trainer.fit(model)

    if args.checkpoint_models:
        print(f'Best model path: {checkpoint_callback.best_model_path}.')
