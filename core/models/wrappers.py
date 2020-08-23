import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict


class BaseModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(BaseModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError('Please use a subclass of BaseModelWrapper.')


class MultilabelClassifierWrapper(BaseModelWrapper):
    """Wraps an embedding model to output a multilabel probabilities for classification.

    Args:
        model (nn.Module): model to produce embeddings
        embedding_size (int): size of output of `model`
        num_labels (int): number of binary categories
    """
    def __init__(self, model: nn.Module, embedding_size: int, num_labels: int):
        super(MultilabelClassifierWrapper, self).__init__(model)
        self.num_labels = num_labels
        self.classifier = nn.Linear(embedding_size, num_labels)

    def forward(self, x):
        embedding = self.model(x)
        output = self.classifier(embedding)
        return torch.sigmoid(output)


class TTAWrapper(BaseModelWrapper):
    """Wraps a model to average predictions over all test-time augmentations of a given batch.

    Args:
        model (nn.Module): model to produce embeddings
    """
    def forward(self, x):
        if len(x.shape) == 3:
            return self.model(x)
        else:
            bs, tta_n, num_channels, signal_length = x.shape
            output = self.model(x.view(-1, num_channels, signal_length)).view(bs, tta_n, -1)
            return output.mean(1)


class MetricModelWrapper(BaseModelWrapper):
    """Wraps a model to return a dictionary of metrics / diagnostic variables for a given batch.

    Args:
        model (nn.Module): model to produce the logits
        metric_functions (Dict[str, Callable]): dictionary of functions that take in the output of
            the model and the ground-truths for the batch and produce a single metric
        activation_function (Callable): callable function to process output of the model. Defaults
            to `torch.sigmoid`.
    """
    def __init__(
        self,
        model: nn.Module,
        metric_functions: Dict[str, Callable],
        activation_function: Callable = torch.sigmoid
    ):
        super(MetricModelWrapper, self).__init__(model)
        self.metric_functions = metric_functions
        self.activation_function = activation_function

    def forward(self, batch):
        x, y = batch
        probabilities = self.activation_function(self.model(x))
        output_metrics = {
            metric_name: self.metric_functions[metric_name](probabilities, y)
            for metric_name in self.metric_functions
        }
        return output_metrics


class MixupWrapper(MetricModelWrapper):
    """Wraps a model to output predictions using mixup / manifold mixup.

    Args:
        model (nn.Module): model to produce embeddings
        metric_functions (Dict[str, Callable]): dictionary of functions that take in the output of
            the model and the ground-truths for the batch and produce a single metric
        activation_function (Callable): callable function to process output of the model. Defaults
            to `torch.sigmoid`.
        alpha (float): mixing factor. Defaults to 0.4
        mixup_layer (int): which layer to do mixup (on the input) on within `model.children()`.
            Defaults to 0. Must be greater than or equal to 0.
    """
    def __init__(
        self,
        model: nn.Module,
        metric_functions: Dict[str, Callable],
        activation_function: Callable = torch.sigmoid,
        alpha: float = 0.4,
        mixup_layer: int = 0
    ):
        sequential_model = nn.Sequential(*[TTAWrapper(child) for child in list(model.children())])
        super(MixupWrapper, self).__init__(
            sequential_model, metric_functions=metric_functions,
            activation_function=activation_function
        )
        self.alpha = alpha
        self.mixup_layer = mixup_layer
        assert self.mixup_layer >= 0, f'mixup_layer must be greater than 0: specified {mixup_layer}'
        assert self.mixup_layer < len(self.model), \
            f'mixup_layer must be less than the number of model layers: specified {mixup_layer}'
        if self.mixup_layer == 0:
            print(
                f'Mixup is being done on the data. If you want to mixup a particular layer input, '
                f'please specify mixup_layer > 0 corresponding to the index of that layer.'
            )

    def forward(self, batch):
        x, y = batch
        if self.training:  # TODO: this takes a long time, profile which step is the bottleneck
            out = x
            mixed_metric_functions = None
            for i, layer in enumerate(self.model):
                if i == self.mixup_layer:
                    out, mixed_y, lam, mixed_metric_functions = self.mixup(out, y, alpha=self.alpha)
                out = layer(out)
            out = self.activation_function(out)
            output_metrics = {}
            for metric_name in mixed_metric_functions:
                try:
                    output_metrics[metric_name] = self.metric_functions[metric_name](out, mixed_y)
                except ValueError:  # In case non-binary targets don't make sense
                    output_metrics[metric_name] = 0.

        else:
            output_metrics = super().forward(batch)

        return output_metrics

    def mixup(self, x, y, alpha=0.4):
        batch_size = x.size()[0]
        if alpha > 0:
            lam = np.random.beta(alpha, alpha, batch_size)
            lam = np.concatenate(
                [lam[:, None], 1 - lam[:, None]], 1
            ).max(1)[:, None, None, None]
            lam = torch.from_numpy(lam).float().to(x.device)
        else:
            lam = 1.
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        lam = lam.view(batch_size, 1)
        y_a, y_b = y, y[index]
        mixed_y = lam * y_a + (1 - lam) * y_b
        mixed_metric_functions = {
            metric_name: (
                lambda y_pred, y_true: (
                    lam * self.metric_functions[metric_name](y_pred, y_a, reduce=False) +
                    (1 - lam) * self.metric_functions[metric_name](y_pred, y_b, reduce=False)
                ).mean()
            )
            for metric_name in self.metric_functions
        }
        return mixed_x, mixed_y, lam, mixed_metric_functions
