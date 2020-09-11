""" Loss functions for training / fine-tuning PTB-XL models. """
from functools import partial
import torch
from torch.nn import functional as F


def loss_function_factory(function_name, **kwargs):
    if function_name == 'bce_loss':
        return bce_loss
    elif function_name == 'f1_loss':
        return f1_loss
    elif function_name == 'focal_loss':
        return partial(
            focal_loss, alpha=kwargs['focal_loss_alpha'], gamma=kwargs['focal_loss_gamma']
        )


def bce_loss(y_pred: torch.tensor, y_true: torch.tensor, reduce: bool = True) -> torch.tensor:
    return F.binary_cross_entropy(
        y_pred.float(), y_true.float(), reduction='mean' if reduce else 'none'
    )


def f1_loss(
    y_pred: torch.tensor, y_true: torch.tensor, epsilon: float = 1e-8, reduce: bool = True
) -> torch.tensor:
    """Soft macro F1-loss for training a classifier.

    Args:
        y_pred (torch.tensor): tensor of predicted probabilities
        y_true (torch.tensor): tensor of binary integer targets
        epsilon (float, optional): padding for any division. Defaults to 1e-8.
        reduce (bool, optional): whether to mean-reduce the loss. Defaults to True.

    Returns:
        torch.tensor: soft macro F1-loss (scalar)
    """
    tp = (y_true * y_pred).sum(axis=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)  # noqa: F841
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    loss = 1 - f1
    if reduce:
        loss = loss.mean()
    return loss


def focal_loss(
    y_pred: torch.tensor,
    y_true: torch.tensor,
    alpha: float = 1,
    gamma: float = 2,
    reduce: bool = True
) -> torch.tensor:
    """Focal loss for training a classifier.

    Args:
        y_pred (torch.tensor): tensor of predicted probabilities
        y_true (torch.tensor): tensor of binary integer targets
        alpha (float, optional): balancing term. Defaults to 1.
        gamma (float, optional): focusing parameter (larger --> downweights easier samples).
            Defaults to 2.
        reduce (bool, optional): whether to mean-reduce the loss. Defaults to True.

    Returns:
        torch.tensor: focal loss (scalar)
    """
    bce_loss_term = F.binary_cross_entropy(y_pred.float(), y_true.float(), reduction='none')
    p_t = torch.exp(-bce_loss_term)
    loss = (alpha * ((1 - p_t) ** gamma) * bce_loss_term)
    if reduce:
        loss = loss.mean()
    return loss
