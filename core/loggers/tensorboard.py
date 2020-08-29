import itertools
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import sklearn.metrics as metrics


class TensorBoardLogger(TensorBoardLogger):
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.experiment.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def log_hyperparams(self, *args, **kwargs):
        if not self.testing:
            super(TensorBoardLogger, self).log_hyperparams(*args, **kwargs)
        else:
            print("Not logging to Tensorboard in test phase.")

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        labeled_figs = confusion_matrix(y_true, y_pred, labels)
        for k, v in labeled_figs.items():
            self.experiment.add_figure(f'Confusion Matrix: {k}', v)
        plt.close('all')

    def plot_roc(self, y_true, y_pred, labels):
        labeled_figs = roc(y_true, y_pred, labels)
        for k, v in labeled_figs.items():
            self.experiment.add_figure(f'ROC Curve: {k}', v)
        plt.close('all')

    def plot_figures(self, figure_dict):
        for k, v in figure_dict.items():
            self.experiment.add_figure(k, v)
        plt.close('all')


def confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cms = metrics.multilabel_confusion_matrix(y_true, y_pred)
    labeled_figs = {}
    for cm, label in zip(cms, classes):
        subfig_classes = [label, 'Rest']
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(f'{title}: {label} vs. Rest')
        plt.colorbar()
        tick_marks = np.arange(len(subfig_classes))
        plt.xticks(tick_marks, subfig_classes, rotation=45)
        plt.yticks(tick_marks, subfig_classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

        plt.tight_layout()
        plt.ylabel('True label')
        labeled_figs[label] = fig
    return labeled_figs


def roc_plot(fpr, tpr, label, color, title, linestyle=None, lw=2):
    fig = plt.figure()
    plt.plot(
        fpr, tpr, label=label, color=color, linestyle=linestyle, linewidth=lw
    )
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return fig


def roc(y_true, y_pred, labels, title='ROC curve'):
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    labeled_figs = {}
    labeled_figs['Micro-Average'] = roc_plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
        title='ROC Curve: Micro-average',
        color='deeppink', linestyle=':', lw=4
    )

    labeled_figs['Macro-Average'] = roc_plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
        title='ROC Curve: Macro-average',
        color='navy', linestyle=':', lw=4
    )

    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        labeled_figs[labels[i]] = roc_plot(
            fpr[i], tpr[i],
            label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:0.2f})',
            title=f'ROC Curve: {labels[i]}',
            color=color, lw=4,
        )

    return labeled_figs
