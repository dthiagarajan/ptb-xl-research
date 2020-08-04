import numpy as np
import torch
from typing import Optional


class ECGDataset(torch.utils.data.Dataset):
    """Dataset for ECG 12-lead data.

    Args:
        signals (np.ndarray): numpy array of signals of shape (number of patients, frequency, leads)
        labels (Optional[np.ndarray]): optional numpy array of binarized labels
        transform (Optional[np.ndarray -> torch.Tensor]): transform for sampling/segmenting the
            signal data
    """
    def __init__(self, signals: np.ndarray, labels: Optional[np.ndarray] = None, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        signal = self.signals[index]
        if self.transform:
            signal = self.transform(signal).float()
        if self.labels is not None:
            return signal, self.labels[index].astype(float)
        else:
            return signal

    def __len__(self):
        return len(self.signals)
