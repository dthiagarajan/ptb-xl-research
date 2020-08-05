import argparse
from functools import partial
import numpy as np
from pathlib import Path
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from core.data.dataset import ECGDataset
from core.data.transforms import (
    binarize_labels,
    window_sampling_transform,
    window_segmenting_test_transform
)
from core.data.utils import load_all_data, split_all_data
import core.models as ptbxl_models
from core.models.wrappers import TTAWrapper


class PTBXLClassificationModel(LightningModule):
    """Lightning module wrapper for training a generic model for PTB-XL classification"""

    allowed_models = [
        'Simple1DCNN',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'wide_resnet50_2', 'wide_resnet101_2'
    ]

    def __init__(self, *args, **kwargs):
        super(PTBXLClassificationModel, self).__init__()
        assert kwargs['model_name'] in self.allowed_models, \
            f"Please pick one of: {self.allowed_models}"
        self.model_name = kwargs['model_name']
        self.task_name = kwargs['task_name']
        self.data_dir = kwargs['data_dir']
        self.batch_size = kwargs['batch_size']
        self.num_input_channels = kwargs['num_input_channels']
        self.num_classes = kwargs['num_classes']
        self.num_workers = kwargs['num_workers']
        self.model = self.initialize_model(
            self.model_name, self.num_input_channels, self.num_classes
        )
        self.sampling_rate = kwargs['sampling_rate']
        self.loss = nn.BCELoss()
        self.save_hyperparameters(*kwargs.keys())

        self.train_step, self.val_step = 0, 0
        self.best_metrics = None

    def initialize_model(self, model_name, num_input_channels, num_classes):
        return TTAWrapper(getattr(ptbxl_models, model_name)(
            num_input_channels=num_input_channels, num_classes=num_classes
        ))

    def update_hyperparams_and_metrics(self, metrics):
        if self.best_metrics is None:
            self.best_metrics = {f'best_{k}': v for (k, v) in metrics.items()}
            self.best_metrics['best_epoch'] = 0
        else:
            flag = True
            for k in metrics:
                if 'loss' in k:
                    if metrics[k] >= self.best_metrics[f'best_{k}']:
                        flag = False
                        break
                elif 'acc' in k:
                    if metrics[k] <= self.best_metrics[f'best_{k}']:
                        flag = False
                        break
            if flag is True:
                self.best_metrics = {f'best_{k}': v for (k, v) in metrics.items()}
                self.best_metrics['best_epoch'] = self.current_epoch
        self.logger.log_hyperparams(self.hparams, self.best_metrics)

    def on_fit_start(self):
        # Need this function to have best metrics being logged in hyperparameters tab of TB
        self.update_hyperparams_and_metrics(
            {'val_epoch_loss': float('inf'), 'val_epoch_acc': 0}
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        probabilities = self(x)
        loss = self.loss(probabilities, y.float())
        acc = ((probabilities > 0.5) == y).sum().float() / (len(y) * len(y[0]))

        tensorboard_logs = {'Train/train_step_loss': loss, 'Train/train_step_acc': acc}
        self.logger.log_metrics(tensorboard_logs, self.train_step)
        self.train_step += 1

        return {'loss': loss, 'acc': acc, 'progress_bar': tensorboard_logs}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        acc_mean = torch.stack([x['acc'] for x in outputs]).mean()
        self.logger.log_metrics(
            {"Train/train_loss": loss_mean, "Train/train_acc": acc_mean}, self.current_epoch + 1
        )
        self.logger.experiment.add_scalars(
            "Losses", {"train_loss": loss_mean}, self.current_epoch + 1
        )
        self.logger.experiment.add_scalars(
            "Accuracies", {"train_acc": acc_mean}, self.current_epoch + 1
        )
        return {'train_epoch_loss': loss_mean, 'train_epoch_acc': acc_mean}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probabilities = self(x)
        loss = self.loss(probabilities, y.float())
        acc = ((probabilities > 0.5) == y).sum().float() / (len(y) * len(y[0]))
        tensorboard_logs = {'Validation/val_step_loss': loss, 'Validation/val_step_acc': acc}
        self.logger.log_metrics(tensorboard_logs, self.val_step)
        self.val_step += 1

        return {'loss': loss, 'acc': acc, 'progress_bar': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['acc'] for x in outputs]).mean()
        self.logger.log_metrics(
            {"Validation/val_loss": val_loss_mean, "Validation/val_acc": val_acc_mean},
            self.current_epoch + 1
        )
        self.logger.experiment.add_scalars(
            "Losses", {"val_loss": val_loss_mean}, self.current_epoch + 1
        )
        self.logger.experiment.add_scalars(
            "Accuracies", {"val_acc": val_acc_mean}, self.current_epoch + 1
        )
        self.update_hyperparams_and_metrics(
            {'val_epoch_loss': val_loss_mean, 'val_epoch_acc': val_acc_mean}
        )
        return {'val_epoch_loss': val_loss_mean, 'val_epoch_acc': val_acc_mean}

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 1e-3},
        ])

    def prepare_data(self):
        if (
            Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_data.npy').exists() and
            Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_labels.npy').exists() and
            Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_data.npy').exists() and
            Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_labels.npy').exists()
        ):
            valid_x_train = np.load(Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_data.npy'))
            valid_y_train = np.load(Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_labels.npy'))
            valid_x_test = np.load(Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_data.npy'))
            valid_y_test = np.load(Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_labels.npy'))

        else:
            X, Y = load_all_data('', self.sampling_rate)
            X_train, y_train, X_test, y_test = split_all_data(X, Y)
            valid_x_train, valid_y_train, mlb = binarize_labels(
                X_train, y_train, self.task_name
            )
            valid_x_test, valid_y_test, _ = binarize_labels(
                X_test, y_test, self.task_name, mlb
            )

            Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name).mkdir(parents=True, exist_ok=True)
            np.save(
                Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_data.npy'), valid_x_train
            )
            np.save(
                Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_labels.npy'), valid_y_train
            )
            np.save(
                Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_data.npy'), valid_x_test
            )
            np.save(
                Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_labels.npy'), valid_y_test
            )

        # Conv1d expects shape (N, C_in, L_in), so we set signal length as the last dimension
        train_transform = transforms.Compose([
            window_sampling_transform(self.sampling_rate * 10, self.sampling_rate * 2),
            np.transpose,  # Setting signal length as the last dimension
            torch.from_numpy
        ])
        test_transform = transforms.Compose([
            window_segmenting_test_transform(self.sampling_rate * 10, self.sampling_rate * 2),
            partial(np.transpose, axes=(0, 2, 1)),  # Setting signal length as the last dimension
            torch.from_numpy
        ])
        self.train_dataset = ECGDataset(valid_x_train, valid_y_train, transform=train_transform)
        self.val_dataset = ECGDataset(valid_x_test, valid_y_test, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='Simple1DCNN')
        parser.add_argument('--task_name', type=str, default='diagnostic_superclass')
        parser.add_argument('--data_dir', type=str, default='./')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_input_channels', type=int, default=12)
        parser.add_argument('--num_classes', type=int, default=5)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--sampling_rate', type=int, default=500)
        return parser