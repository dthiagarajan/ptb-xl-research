from functools import partial
import numpy as np
import pytest
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from core.data.dataset import ECGDataset
from core.data.transforms import (
    binarize_labels,
    window_sampling_transform,
    window_segmenting_test_transform
)
from core.data.utils import load_all_data, split_all_data
from core.models import (
    resnet18,
    Simple1DCNN
)
from core.models.wrappers import MultilabelClassifierWrapper, TTAWrapper
from .conftest import infer, train


@pytest.fixture
def benchmarks():
    return {
        'max_train_loss': 0.02,
        'min_train_accuracy': 0.9,
        'max_test_loss': 0.02,
        'min_test_accuracy': 0.75
    }


@pytest.fixture
def dataloaders():
    torch.manual_seed(31)
    np.random.seed(31)
    random.seed(31)

    X, Y = load_all_data('', 100)
    X_train, y_train, X_test, y_test = split_all_data(X, Y)
    valid_x_train, valid_y_train, mlb = binarize_labels(
        X_train, y_train, 'diagnostic_superclass'
    )
    valid_x_train, valid_y_train = valid_x_train[:320], valid_y_train[:320]
    valid_x_test, valid_y_test, _ = binarize_labels(
        X_test, y_test, 'diagnostic_superclass', mlb
    )
    valid_x_test, valid_y_test = valid_x_test[:320], valid_y_test[:320]

    # Conv1d expects shape (N, C_in, L_in), so we set signal length as the last dimension
    train_transform = transforms.Compose([
        window_sampling_transform(1000, 100),
        np.transpose,  # Setting signal length as the last dimension
        torch.from_numpy
    ])
    test_transform = transforms.Compose([
        window_segmenting_test_transform(1000, 100),
        partial(np.transpose, axes=(0, 2, 1)),  # Setting signal length as the last dimension
        torch.from_numpy
    ])

    train_ds = ECGDataset(valid_x_train, valid_y_train, transform=train_transform)
    test_ds = ECGDataset(valid_x_test, valid_y_test, transform=test_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=True)
    return train_dl, test_dl


class TestSimple1DCNN():
    @pytest.mark.functional
    def test_training_and_eval_simple_1d_cnn(self, dataloaders, benchmarks):
        train_dl, test_dl = dataloaders
        clf = TTAWrapper(MultilabelClassifierWrapper(Simple1DCNN(), 128, 5))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
        for epoch in range(10):
            train_epoch_loss, train_epoch_accuracy = train(clf, train_dl, criterion, optimizer)
            print(
                f'Training epoch {epoch}: Loss [{train_epoch_loss:.3f}], '
                f'Accuracy [{train_epoch_accuracy:.3f}]'
            )
            test_epoch_loss, test_epoch_accuracy = infer(clf, test_dl, criterion)
            print(
                f'Testing epoch {epoch}: Loss [{test_epoch_loss:.3f}], '
                f'Accuracy [{test_epoch_accuracy:.3f}]'
            )
        assert train_epoch_loss <= benchmarks['max_train_loss']
        assert train_epoch_accuracy >= benchmarks['min_train_accuracy']
        assert test_epoch_loss <= benchmarks['max_test_loss']
        assert test_epoch_accuracy >= benchmarks['min_test_accuracy']


class TestResNet18():
    @pytest.mark.functional
    def test_training_and_eval_simple_1d_cnn(self, dataloaders, benchmarks):
        train_dl, test_dl = dataloaders
        clf = TTAWrapper(MultilabelClassifierWrapper(
            resnet18(num_input_channels=12, num_classes=128), 128, 5)
        )
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
        for epoch in range(10):
            train_epoch_loss, train_epoch_accuracy = train(clf, train_dl, criterion, optimizer)
            print(
                f'Training epoch {epoch}: Loss [{train_epoch_loss:.3f}], '
                f'Accuracy [{train_epoch_accuracy:.3f}]'
            )
            test_epoch_loss, test_epoch_accuracy = infer(clf, test_dl, criterion)
            print(
                f'Testing epoch {epoch}: Loss [{test_epoch_loss:.3f}], '
                f'Accuracy [{test_epoch_accuracy:.3f}]'
            )
        assert train_epoch_loss <= benchmarks['max_train_loss']
        assert train_epoch_accuracy >= benchmarks['min_train_accuracy']
        assert test_epoch_loss <= benchmarks['max_test_loss']
        assert test_epoch_accuracy >= benchmarks['min_test_accuracy']
