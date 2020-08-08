import numpy as np

from core.metrics import (
    AUC,
    metric_summary,
    multi_threshold_precision_recall,
)


def test_auc():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert AUC(y_true, y_scores) == 0.75


def test_metric_summary():
    y_true = np.array([[0., 1., 1., 1.], [1., 0., 0., 0.]])
    y_scores = np.array([[0.1, 0.6, 0.75, 0.8], [0.1, 0.6, 0.75, 0.8]])
    f_max, auc, f_scores, average_precisions, average_recalls, thresholds = metric_summary(
        y_true, y_scores
    )
    assert len(f_scores) == 10 and len(average_precisions) == 10 and len(average_recalls) == 10
    assert f_max == 2/3
    assert auc == 0.5
    assert (f_scores == [2/3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.25, np.nan, np.nan])[:-2].all()
    assert (average_precisions == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan, np.nan])[:-2].all()
    assert (average_recalls == [1., 0.5, 0.5, 0.5, 0.5, 0.5, 1/3, 1/6, 0., 0.]).all()


def test_multi_threshold_precision_recall():
    y_true = np.array([0, 0, 1, 1])[None, :]
    y_scores = np.array([0.1, 0.6, 0.75, 0.8])[None, :]
    thresholds = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    average_precisions, average_recalls = multi_threshold_precision_recall(
        y_true, y_scores, thresholds
    )
    assert (average_precisions == [2/3, 2/3, 2/3, 2/3, 2/3, 1., 1.]).all()
    assert (average_recalls == [1, 1, 1, 1, 1, 1, 0.5]).all()
