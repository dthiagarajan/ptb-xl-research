import numpy as np
from typing import List


def flatten_signal(segmented_signal: np.ndarray) -> List[np.ndarray]:
    """Flattens a segmented signal into the original signal.

    Args:
        segmented_signal (np.ndarray): signal pieces of shape (P, L, W)

    Returns:
        List[np.ndarray]: the original L signals of length ((P+1) * W / 2)
    """
    flattened_signals = [
        np.hstack([
            hs[:len(hs)//2] if i < len(segmented_signal) - 1 else hs
            for i, hs in enumerate(segmented_signal[:, i, :])
        ])
        for i in range(segmented_signal.shape[1])
    ]
    return flattened_signals


def flatten_heatmap(heatmap: np.ndarray):
    """Flattens a heatmap for each piece of a segmented signal into a single signal's heatmap.

    Args:
        heatmap (np.ndarray): heatmap pieces of shape (P, W)

    Returns:
        np.ndarray: a single heatmap for all leads of length ((P+1) * W / 2)
    """
    flattened_heatmap = np.hstack([
        hs[:len(hs)//2] if i < len(heatmap) - 1 else hs
        for i, hs in enumerate(heatmap)
    ])
    return flattened_heatmap
