""" Loss functions for training / fine-tuning PTB-XL models. """
from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F


def loss_function_factory(function_name, **kwargs):
    if function_name == 'bce_loss':
        return partial(
            bce_loss,
            label_weight_mapping=(
                kwargs['label_weight_mapping']
                if 'class_weighted_loss' in kwargs and kwargs['class_weighted_loss'] is True
                else None
            )
        )
    elif function_name == 'f1_loss':
        return f1_loss
    elif function_name == 'focal_loss':
        return partial(
            focal_loss, alpha=kwargs['focal_loss_alpha'], gamma=kwargs['focal_loss_gamma']
        )


def bce_loss(
    y_pred: torch.tensor,
    y_true: torch.tensor,
    reduce: bool = True,
    label_weight_mapping: dict = None
) -> torch.tensor:
    weight = None
    if label_weight_mapping is not None:
        weight = torch.tensor([
            [label_weight_mapping[col][int(t[col].item())] for col in range(len(t))]
            for t in y_true
        ])
        if torch.cuda.is_available():
            weight = weight.cuda()
    loss = F.binary_cross_entropy(y_pred.float(), y_true.float(), weight=weight)
    if reduce:
        loss = loss.mean()
    return loss


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


class NTXEntCriterion(nn.Module):
    """Normalized, temperature-scaled cross-entropy criterion, as suggested in the SimCLR paper.
    Parameters:
        temperature (float, optional): temperature to scale the confidences. Defaults to 0.5.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    similarity = nn.CosineSimilarity(dim=2)

    def __init__(self, temperature=0.5):
        super(NTXEntCriterion, self).__init__()
        self.temperature = temperature
        self.batch_size = None
        self.mask = None

    def mask_correlated_samples(self, batch_size):
        """Masks examples in a batch and it's augmented pair for computing the valid summands for
            the criterion.
        Args:
            batch_size (int): batch size of the individual batch (not including it's augmented pair)
        Returns:
            torch.Tensor: a mask (tensor of 0s and 1s), where 1s indicates a pair of examples in a
                batch that will contribute to the overall batch loss
        """
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def compute_similarities(self, z_i, z_j, temperature):
        """Computes the similarities between two projections `z_i` and `z_j`, scaling based on
            `temperature`.
        Args:
            z_i (torch.Tensor): projection of a batch
            z_j (torch.Tensor): projection of the augmented pair for the batch
            temperature (float): temperature to scale the similarity by
        Returns:
            torch.Tensor: tensor of similarities for the positive and negative pairs
        """
        batch_size = len(z_i)
        mask = self.mask_correlated_samples(batch_size)

        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity(p1.unsqueeze(1), p1.unsqueeze(0)) / temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            batch_size * 2, 1
        )
        negative_samples = sim[mask].reshape(batch_size * 2, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits

    def compute_activations(self, z):
        """Computes activations, i.e. similarities, for the batch and its augmented pair.

        Args:
            z (torch.Tensor): tensor of a batch and its augmented pair, concatenated
        Returns:
            torch.Tensor: activations (similarities) for the given batch
        """
        double_batch_size = len(z)
        batch_size = double_batch_size // 2
        z_i, z_j = z[:double_batch_size // 2], z[double_batch_size // 2:]
        if self.batch_size is None or batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = None

        if self.mask is None:
            self.mask = self.mask_correlated_samples(self.batch_size)

        logits = self.compute_similarities(z_i, z_j, self.temperature)
        return logits

    def forward(self, logits, labels):
        """Computes the loss for a batch and its augmented pair.
        Args:
            logits (torch.Tensor): tensor of logits (similarities) for batch pairs
            labels (torch.Tensor): unsupervised labels for batch pairs
        Returns:
            torch.Tensor: loss for the given batch
        """
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss
