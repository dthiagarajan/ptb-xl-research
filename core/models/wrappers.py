import torch
import torch.nn as nn


class MultilabelClassifierWrapper(nn.Module):
    """Wraps an embedding model to output a multilabel probabilities for classification.

    Args:
        model (nn.Module): model to produce embeddings
        embedding_size (int): size of output of `model`
        num_labels (int): number of binary categories
    """
    def __init__(self, model: nn.Module, embedding_size: int, num_labels: int):
        super(MultilabelClassifierWrapper, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.classifier = nn.Linear(embedding_size, num_labels)

    def forward(self, x):
        embedding = self.model(x)
        output = self.classifier(embedding)
        return torch.sigmoid(output)


class TTAWrapper(nn.Module):
    """Wraps a model to average predictions over all test-time augmentations of a given batch.

    Args:
        model (nn.Module): model to produce embeddings
    """
    def __init__(self, model: nn.Module):
        super(TTAWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        if len(x.shape) == 3:
            return self.model(x)
        else:
            bs, tta_n, num_channels, signal_length = x.shape
            output = self.model(x.view(-1, num_channels, signal_length)).view(bs, tta_n, -1)
            return output.mean(1)
