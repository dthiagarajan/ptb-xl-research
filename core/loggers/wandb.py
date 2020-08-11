import itertools
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import sklearn.metrics as metrics
import wandb
from wandb.plots import ROC


class WandbLogger(WandbLogger):
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        metrics = {f'{main_tag}/{k}': v for k, v in tag_scalar_dict.items()}
        if global_step is not None:
            metrics['epoch'] = global_step
        self.log_metrics(metrics)

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        wandb.log(confusion_matrix(y_true, y_pred, labels))

    def plot_roc(self, y_true, y_pred, labels):
        wandb.log({
            f'roc_{label}': ROC(
                y_true[:, i], np.column_stack([1 - y_pred[:, i], y_pred[:, i]]), ['Rest', label],
                classes_to_plot=[1]
            )
            for i, label in enumerate(labels)
        })

    def plot_figures(self, figure_dict):
        wandb.log({k: wandb.Image(v) for k, v in figure_dict.items()})
        plt.close('all')


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray):
    """Computes the multilabel confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true (np.ndarray): list of targets
        y_pred (np.ndarray): list of (discrete) predictions
        labels (np.ndarray): list of class names

    Returns:
        wandb.visualize object for plotting a confusion matrix that can be passed to wandb.log
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cms = metrics.multilabel_confusion_matrix(y_true, y_pred)

    def confusion_matrix_table(cm, label):
        data = []
        count = 0
        pred_classes, true_classes = [label, 'Rest'], [label, 'Rest']
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            pred_dict = pred_classes[i]
            true_dict = true_classes[j]
            data.append([pred_dict, true_dict, cm[i, j]])
            count += 1
            if count >= wandb.Table.MAX_ROWS:
                wandb.termwarn(
                    f"wandb uses only the first {wandb.Table.MAX_ROWS} datapoints to create plots."
                )
                break
        return wandb.visualize(
            'wandb/confusion_matrix/v1',
            wandb.Table(columns=['Predicted', 'Actual', 'Count'], data=data)
        )

    return {
        f'{label}_confusion_matrix': confusion_matrix_table(cms[i], label)
        for i, label in enumerate(labels)
    }
