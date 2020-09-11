import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple
import warnings


def multi_threshold_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes precision and recall for multiple thresholds.

    Args:
        y_true (np.ndarray): list of labels
        y_pred (np.ndarray): list of predictions (probabilities)
        thresholds (np.ndarray): list of thresholds to use on predicted probabilities

    Returns:
        Tuple[np.ndarray, np.ndarray]: average precision and recall over all examples,
            for each threshold

    Reference:
    https://github.com/helme/ecg_ptbxl_benchmarking/blob/98eda569affe6a44d9ea7bbb34b57cc1e2862d03/code/utils/utils.py#L91
    """
    # Expand analysis to number of thresholds
    y_pred_bin = np.repeat(y_pred[None, :, :], len(thresholds), axis=0) >= thresholds[:, None, None]
    y_true_bin = np.repeat(y_true[None, :, :], len(thresholds), axis=0)

    # Compute true positives
    TP = np.sum(np.logical_and(y_true, y_pred_bin), axis=2)

    # Compute macro-average precision handling all warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        den = np.sum(y_pred_bin, axis=2)
        precision = TP / den
        precision[den == 0] = np.nan
        with warnings.catch_warnings():  # for nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_precision = np.nanmean(precision, axis=1)

    # Compute macro-average recall
    recall = TP / np.sum(y_true_bin, axis=2)
    av_recall = np.mean(recall, axis=1)

    return av_precision, av_recall


def metric_summary(
    y_true: np.ndarray, y_pred: np.ndarray, num_thresholds: int = 10
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Returns a summary of metrics across `num_thresholds` thresholds.

    Args:
        y_true (np.ndarray): list of labels
        y_pred (np.ndarray): list of predicted probabilities
        num_thresholds (int, optional): number of desired thresholds. Defaults to 10.

    Returns:
        float: max F1-score across all thresholds
        float: AUC score
        np.ndarray: F1-scores for each threshold based on average precision and recalls
        np.ndarray: average precision for each threshold
        np.ndarray: average recall for each threshold
        np.ndarray: thresholds used
    """
    thresholds = np.arange(0.00, 1.01, 1. / (num_thresholds - 1), float)
    average_precisions, average_recalls = multi_threshold_precision_recall(
        y_true, y_pred, thresholds
    )
    f_scores = 2 * (average_precisions * average_recalls) / (average_precisions + average_recalls)
    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        print(f'Value error encountered, likely due to using mixup. Setting AUC to 0.')
        auc = 0.
    return (
        f_scores[np.nanargmax(f_scores)],
        auc,
        f_scores,
        average_precisions,
        average_recalls,
        thresholds
    )


def AUC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the macro-average AUC score.

    Args:
        y_true (np.ndarray): list of labels
        y_pred (np.ndarray): list of predicted probabilities

    Returns:
        float: macro-average AUC score.
    """
    return roc_auc_score(y_true, y_pred, average='macro')
