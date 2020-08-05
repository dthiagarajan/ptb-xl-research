import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MultiLabelBinarizer

from core.data.transforms import (
    binarize_labels,
    window_sampling_transform,
    window_segmenting_test_transform
)


@pytest.fixture
def label_series():
    return pd.Series(
        [[1, 2, 3], [1, 2], [1], [2], [3]]
    )


@pytest.fixture
def trained_mlb():
    mlb = MultiLabelBinarizer()
    mlb.fit([(1, 2), (3,)])
    return mlb


class TestBinarizeLabel:
    def test_binarize_train_labels(self, label_series):
        data = np.random.randn(len(label_series), 10)
        valid_data, valid_labels, mlb = binarize_labels(data, label_series, 'label')
        expected_result = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        assert len(valid_data) == len(valid_labels) + 1
        assert (valid_labels == expected_result).all()
        assert (mlb.classes_ == np.array([1, 2, 3])).all()

    def test_binarize_test_labels(self, label_series, trained_mlb):
        data = np.random.randn(len(label_series), 10)
        valid_data, valid_labels, mlb = binarize_labels(
            data, label_series, 'label', mlb=trained_mlb
        )
        expected_result = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        assert len(valid_data) == len(valid_labels)
        assert (valid_labels == expected_result).all()
        assert (mlb.classes_ == np.array([1, 2, 3])).all()


class TestWindowSamplingTransform:
    @pytest.mark.parametrize('signal_length,window_size', [(10, 4), (10, 5), (10, 10)])
    def test_window_sampling_transform(self, signal_length, window_size):
        transform = window_sampling_transform(signal_length, window_size)
        signal = np.random.randn(signal_length)
        assert len(transform(signal)) == window_size


class TestWindowSegmentingTransform:
    @pytest.mark.parametrize('signal_length,window_size', [(10, 4), (10, 5), (10, 10)])
    def test_window_segmenting_test_transform(self, signal_length, window_size):
        transform = window_segmenting_test_transform(signal_length, window_size)
        signal = np.random.randn(signal_length)

        window_starts = set()
        # Take all overlapping windows of size `window_size` from 0 to end
        for index in range(0, signal_length, window_size // 2):
            if index + window_size <= signal_length:
                window_starts.add(index)
            else:
                break

        # Take all overlapping windows of size `window_size` from end to 0
        for index in range(signal_length - window_size, 0, -window_size // 2):
            if index + window_size <= signal_length:
                window_starts.add(index)
            else:
                break

        assert len(transform(signal)) == len(window_starts)
