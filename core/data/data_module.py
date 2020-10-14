import argparse
from functools import partial
import numpy as np
from pathlib import Path
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Optional

from core.data.dataset import ECGDataset
from core.data.transforms import (
    binarize_labels,
    window_sampling_transform,
    window_segmenting_test_transform
)
from core.data.utils import load_all_data, split_all_data


class PTBXLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, sampling_rate, task_name, batch_size=128, num_workers=0):
        super(PTBXLDataModule, self).__init__()
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.task_name = task_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        try:
            self.get_label_weight_mapping()
        except AttributeError:
            self.setup()

    def setup(self, stage: Optional[str] = None):
        train_data_path = Path(
            self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_data.npy'
        )
        train_labels_path = Path(
            self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'train_labels.npy'
        )
        test_data_path = Path(
            self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_data.npy'
        )
        test_labels_path = Path(
            self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'test_labels.npy'
        )
        label_names_path = Path(
            self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'label_names.npy'
        )
        if (
            train_data_path.exists() and train_labels_path.exists() and test_data_path.exists() and
            test_labels_path.exists() and label_names_path.exists()
        ):
            valid_x_train = np.load(train_data_path)
            valid_y_train = np.load(train_labels_path)
            valid_x_test = np.load(test_data_path)
            valid_y_test = np.load(test_labels_path)
            self.labels = np.load(label_names_path, allow_pickle=True)

        else:
            X, Y = load_all_data(self.data_dir, self.sampling_rate, task_name=self.task_name)
            X_train, y_train, X_test, y_test = split_all_data(X, Y, task_name=self.task_name)
            valid_x_train, valid_y_train, mlb = binarize_labels(
                X_train, y_train, self.task_name
            )
            valid_x_test, valid_y_test, _ = binarize_labels(
                X_test, y_test, self.task_name, mlb
            )
            self.labels = mlb.classes_

            Path(self.data_dir, 'data', str(self.sampling_rate), self.task_name).mkdir(
                parents=True, exist_ok=True
            )
            np.save(train_data_path, valid_x_train)
            np.save(train_labels_path, valid_y_train)
            np.save(test_data_path, valid_x_test)
            np.save(test_labels_path, valid_y_test)
            np.save(label_names_path, mlb.classes_)

        # Conv1d expects shape (N, C_in, L_in), so we set signal length as the last dimension
        train_transform = transforms.Compose([
            window_sampling_transform(self.sampling_rate * 10, int(self.sampling_rate * 2.5)),
            np.transpose,  # Setting signal length as the last dimension
            torch.from_numpy
        ])
        test_transform = transforms.Compose([
            window_segmenting_test_transform(self.sampling_rate * 10, int(self.sampling_rate * 2.5)),
            partial(np.transpose, axes=(0, 2, 1)),  # Setting signal length as the last dimension
            torch.from_numpy
        ])
        self.train_dataset = ECGDataset(valid_x_train, valid_y_train, transform=train_transform)
        self.val_dataset = ECGDataset(valid_x_test, valid_y_test, transform=test_transform)

        self.get_label_weight_mapping()

    def get_label_weight_mapping(self):
        class_counts_fp = Path(
            self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'class_counts.pkl'
        )
        class_weights_fp = Path(
            self.data_dir, 'data', str(self.sampling_rate), self.task_name, 'class_weights.pkl'
        )
        if class_counts_fp.exists() and class_weights_fp.exists():
            with open(class_counts_fp, 'rb') as f:
                self.label_counts_mapping = pickle.load(f)
            with open(class_weights_fp, 'rb') as f:
                self.label_weight_mapping = pickle.load(f)
        else:
            train_labels = np.stack([it[1] for it in self.train_dataset])
            self.label_counts_mapping = {
                col: np.unique(train_labels[:, col], return_counts=True)[1]
                for col in range(train_labels.shape[1])
            }
            label_weight_mapping = {
                col: (self.label_counts_mapping[col].sum() / self.label_counts_mapping[col])
                for col in range(train_labels.shape[1])
            }
            self.label_weight_mapping = {
                col: label_weight_mapping[col] / label_weight_mapping[col].sum()
                for col in range(train_labels.shape[1])
            }
            with open(class_counts_fp, 'wb') as f:
                pickle.dump(self.label_counts_mapping, f)
            with open(class_weights_fp, 'wb') as f:
                pickle.dump(self.label_weight_mapping, f)

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

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--task_name', type=str, default='diagnostic_superclass')
        parser.add_argument('--data_dir', type=str, default='./')
        parser.add_argument('--sampling_rate', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=0)
        return parser
