import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict

from core.metrics import AUC
from core.models import loss_functions


class BaseModelCallback:
    def on_forward_start(self, module, batch, batch_idx, model_output=None):
        return model_output

    def on_forward_end(self, module, batch, batch_idx, model_output=None):
        return model_output


class TTACallback(BaseModelCallback):
    """Wraps a model to average predictions over all test-time augmentations of a given batch."""
    def __init__(self):
        self.tta_applied = False

    def on_forward_start(self, module, batch, batch_idx, model_output=None):
        x, y = module.next_batch
        if len(x.shape) == 4:
            self.tta_applied = True
            bs, tta_n, num_channels, signal_length = x.shape
            module.next_batch = (x.view(-1, num_channels, signal_length), y)
            module.bs, module.tta_n = bs, tta_n

    def on_forward_end(self, module, batch, batch_idx, model_output=None):
        if self.tta_applied:
            model_output = model_output.view(module.bs, module.tta_n, -1).max(1)[0]
            self.tta_applied = False
        return model_output


class LossComposition:
    def __init__(self, loss_component_functions):
        self.loss_component_functions = loss_component_functions

    def __call__(self, y_pred, y_true):
        return sum([
            fac * slcf(y_pred, y_true) for fac, slcf in self.loss_component_functions
        ])


class LossCallback(BaseModelCallback):
    """Wraps a model to call an activation and loss function on the model output."""

    allowed_loss_function_components = ['bce_loss', 'f1_loss', 'focal_loss']

    def __init__(self, loss_function: Callable, activation_function: Callable = torch.sigmoid):
        self.loss_function = loss_function
        self.activation_function = activation_function

    def on_forward_end(self, module, batch, batch_idx, model_output=None):
        _, y = module.next_batch
        probabilities = self.activation_function(model_output)
        model_output = (probabilities, self.loss_function(probabilities, y))
        return model_output

    @classmethod
    def initialize_loss_function(cls, loss_function='bce_loss', **kwargs):
        loss_component_functions = []
        for loss_component in loss_function.split('+'):
            subloss_components = [slc.strip() for slc in loss_component.strip().split('*')]
            assert len(subloss_components) <= 2, \
                f'Cannot specify more than 2 factors (raised on {loss_component}).'
            float_factor, subloss_func = 1.0, None
            for factor in subloss_components:
                try:
                    float_factor = float(factor)
                except ValueError:
                    assert factor in cls.allowed_loss_function_components, \
                        f'Loss function term must be one of {cls.allowed_loss_function_components}'
                    subloss_func = loss_functions.loss_function_factory(factor, **kwargs)

            loss_component_functions.append((float_factor, subloss_func))

        return LossComposition(loss_component_functions)


def acc(y_pred, y_true):
    return ((y_pred > 0.5) == y_true).sum().float() / (len(y_true) * len(y_true[0]))


def auc(y_pred, y_true):
    return torch.Tensor([AUC(y_true.cpu().numpy(), y_pred.detach().cpu().numpy())])


def probs_identity(y_pred, y_true):
    return y_pred


def targets_identity(y_pred, y_true):
    return y_true


class MetricOutputCallback(BaseModelCallback):
    """Wraps a model to return a dictionary of metrics / diagnostic variables for a given batch."""
    def __init__(self, metric_functions: Dict[str, Callable]):
        super(MetricOutputCallback, self).__init__()
        self.metric_functions = metric_functions

    def on_forward_end(self, module, batch, batch_idx, model_output=None):
        _, y = module.next_batch
        probabilities, loss = model_output
        model_output = {}
        for metric_name in self.metric_functions:
            try:
                model_output[metric_name] = self.metric_functions[metric_name](probabilities, y)
            except ValueError:
                # e.g. during mixup, AUC won't make sense
                model_output[metric_name] = torch.tensor(0.)
        model_output['loss'] = loss
        return model_output

    @classmethod
    def initialize_metric_functions(cls):
        return {
            'acc': acc,
            'auc': auc,
            'probs': probs_identity,
            'targets': targets_identity
        }


""" Classes of modules that should be avoided when using mixup.
    Mostly modules that are propagating inputs and models with recurrent layers. """
non_mixable_module_types = [
    nn.Sequential, nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
    nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell, nn.RNN, nn.RNNBase, nn.RNNCell, nn.RNNCellBase
]


class MixupCallback(MetricOutputCallback):
    def __init__(
        self,
        metric_functions: Dict[str, Callable],
        mixup_alpha: float = 0.4,
        mixup_layer: int = 0
    ):
        super(MixupCallback, self).__init__(metric_functions)
        self.module_list = None
        self.mixup_alpha = mixup_alpha
        self.mixup_layer = mixup_layer

    def _is_mixable(self, m):
        "Checks wether the module m is an instance of a module that is allowed for mixup."
        return not any(
            isinstance(m, non_mixable_class) for non_mixable_class in non_mixable_module_types
        )

    def _is_block_module(self, m):
        "Checks whether a module is a Block or Bottleneck"
        m = str(type(m)).lower()
        return "block" in m or "bottleneck" in m

    def _get_mixup_module_list(self, model):
        module_list = list(model.modules())
        # Checks for blocks in the network modules
        block_modules = list(filter(self._is_block_module, module_list))
        if len(block_modules) != 0:
            print(
                f'Mixup callback: Block structure detected, {len(block_modules)} '
                f'modules will be used for mixup.'
            )
            return block_modules

        # Checks for any module that is mixable
        mixable_modules = list(filter(self._is_mixable, module_list))
        if len(mixable_modules) != 0:
            print(
                f'Mixup callback: no known network structure detected, '
                f'{len(mixable_modules)} modules will be used for mixup.'
            )
            return mixable_modules

    def sample_mixing_params(self, batch_size):
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size)
            lam = np.concatenate(
                [lam[:, None], 1 - lam[:, None]], 1
            ).max(1)[:, None, None, None]
            lam = torch.from_numpy(lam).double()
            if torch.cuda.is_available():
                lam = lam.cuda()
        else:
            lam = 1.
        index = torch.randperm(batch_size)
        if torch.cuda.is_available():
            index = index.cuda()
        return lam, index

    def mixup(self, output):
        view_size = np.array(output.shape)
        view_size[1:] = 1
        view_size[-1] = -1
        lam = self.lam.view(*view_size)
        mixed_output = lam * output + (1 - lam) * output[self.index_shuffle]
        return mixed_output.float()

    def hook_mixup(self, module, input, output):
        mixed_output = self.mixup(output)
        return mixed_output

    def on_forward_start(self, module, batch, batch_idx, model_output=None):
        if self.module_list is None:
            self.module_list = self._get_mixup_module_list(module.model)
        if module.training:
            x, y = module.next_batch
            self.lam, self.index_shuffle = self.sample_mixing_params(y.size()[0])
            module.next_batch = (x, self.mixup(y))
            self.mixup_hook_handle = self.module_list[self.mixup_layer].register_forward_hook(
                self.hook_mixup
            )

    def on_forward_end(self, module, batch, batch_idx, model_output=None):
        model_output = super(MixupCallback, self).on_forward_end(
            module, batch, batch_idx, model_output=model_output
        )
        if module.training:
            self.mixup_hook_handle.remove()
        return model_output


def callback_factory(**kwargs):
    return [
        TTACallback(),
        LossCallback(LossCallback.initialize_loss_function(**kwargs)),
        (
            MixupCallback(
                MixupCallback.initialize_metric_functions(),
                kwargs['mixup_alpha'],
                kwargs['mixup_layer']
            ) if kwargs['mixup'] else
            MetricOutputCallback(MetricOutputCallback.initialize_metric_functions())
        ),
    ]
