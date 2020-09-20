import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Optional, Tuple


class WindowSampler:
    def __init__(self, window_size: int, signal_length: int):
        self.window_size = window_size
        self.signal_length = signal_length

    def __call__(self, signal: np.ndarray):
        assert len(signal) == self.signal_length
        # Sample an index where window starting at that number can fit
        index = np.random.randint(0, self.signal_length - self.window_size + 1)
        return signal[index:index + self.window_size]


def window_sampling_transform(signal_length: int, window_size: int):
    """Returns a transform function for sampling a window of size `window_size` from a signal of
    length `signal_length`.

    Args:
        signal_length (int): length of signal to be sampled from
        window_size (int): size of desired window

    Returns:
        np.ndarray -> np.ndarray: function to sample a window of size `window_size` from a signal
            of length `signal_length`
    """
    assert window_size <= signal_length
    return WindowSampler(window_size, signal_length)


class WindowSegmenter:
    def __init__(self, window_starts: List[int], window_size: int, signal_length: int):
        self.window_starts = window_starts
        self.window_size = window_size
        self.signal_length = signal_length

    def __call__(self, signal: np.ndarray):
        assert len(signal) == self.signal_length
        return np.stack([
            signal[index:index+self.window_size] for index in self.window_starts
        ])


def window_segmenting_test_transform(signal_length: int, window_size: int):
    """Returns a transform function for sampling a window of size `window_size` from a signal of
    length `signal_length`.

    Args:
        signal_length (int): length of signal to be sampled from
        window_size (int): size of desired window

    Returns:
        np.ndarray -> np.ndarray: function to return all `window_size` segments from a signal of
            length `signal_length`, with overlap of half `window_size`

    Note:
        this returns all such windows going forward and going backward

    Example:
    >>> window_segmenting_test_transform(6, 4)(np.array([0, 1, 2, 3, 4, 5]))
    array([[0, 1, 2, 3], [2, 3, 4, 5]])
    """
    assert window_size <= signal_length
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

    return WindowSegmenter(window_starts, window_size, signal_length)


def binarize_labels(
    data: np.ndarray,
    label_series: pd.Series,
    label_name: str,
    mlb: Optional[MultiLabelBinarizer] = None
) -> Tuple[np.ndarray, np.ndarray, MultiLabelBinarizer]:
    label_df = pd.DataFrame({label_name: label_series})
    label_df[f'{label_name}_len'] = label_df[label_name].apply(lambda x: len(x))

    valid_x = data[label_df[f'{label_name}_len'] > 0]
    valid_y = label_df[label_df[f'{label_name}_len'] > 0]

    if mlb:
        valid_y_binarized = mlb.transform(valid_y[label_name].values)
    else:
        mlb = MultiLabelBinarizer()
        valid_y_binarized = mlb.fit_transform(valid_y[label_name].values)

    return valid_x, valid_y_binarized, mlb
