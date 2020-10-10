""" Shows heatmaps for a trained a classification model on the PTB-XL dataset.
Example usage:
python show_heatmaps.py --model_name Simple1DCNN --fast_dev_run True
python show_heatmaps.py --model_name resnet18 --fast_dev_run True

To run with a single GPU:
python show_heatmaps.py --model_name resnet18 --fast_dev_run True --gpus 1

To generate multiple layers' heatmaps:
python show_heatmaps.py --model_name Simple1DCNN --heatmap_layers pool1 pool2 pool3
"""
import argparse
from pytorch_lightning import seed_everything, Trainer

from core.data import PTBXLDataModule
from core.loggers import TensorBoardLogger, WandbLogger
from core.models import PTBXLClassificationModel


def get_args():
    """Argument parser for showing heatmaps for a trained PTB-XL classification model.

    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Train a classification model on PTB-XL diagnostic superclass'
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
    model.show_heatmaps = True
    print(f'...done.')
    data_module = PTBXLDataModule(
        args.data_dir, args.sampling_rate, args.task_name,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    data_module.setup()

    if args.logger_platform == 'wandb':
        logger = WandbLogger(project="ptb-xl")
    elif args.logger_platform == 'tensorboard':
        logger = TensorBoardLogger('./heatmap_logs', name='')
        model.log_dir = './heatmap_logs'

    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=logger)

    model.labels = data_module.labels  # Needed for confusion matrix labels
    trainer.test(model=model, datamodule=data_module)
