from abc import abstractmethod
from copy import deepcopy

import torch
from torch import nn as nn
from typing import Any, Callable, List, Tuple

from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.utilities import rank_zero_warn, move_data_to_device
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class VerificationBase:
    """
    Base class for model verification.
    All verifications should run with any :class:`torch.nn.Module` unless otherwise stated.
    """

    def __init__(self, model: nn.Module):
        """
        Arguments:
            model: The model to run verification for.
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def check(self, *args, **kwargs) -> bool:
        """ Runs the actual test on the model. All verification classes must implement this.
        Arguments:
            *args: Any positional arguments that are needed to run the test
            *kwargs: Keyword arguments that are needed to run the test
        Returns:
            `True` if the test passes, and `False` otherwise. Some verifications can only be performed
            with a heuristic accuracy, thus the return value may not always reflect the true state of
            the system in these cases.
        """
        pass

    def _get_input_array_copy(self, input_array=None) -> Any:
        """
        Returns a deep copy of the example input array in cases where it is expected that the
        input changes during the verification process.
        Arguments:
            input_array: The input to clone.
        """
        if input_array is None and isinstance(self.model, LightningModule):
            input_array = self.model.example_input_array
        input_array = deepcopy(input_array)

        if isinstance(self.model, LightningModule):
            input_array = self.model.transfer_batch_to_device(
                input_array, self.model.device
            )
        else:
            input_array = move_data_to_device(
                input_array, device=next(self.model.parameters()).device
            )

        return input_array

    def _model_forward(self, input_array: Any) -> Any:
        """
        Feeds the input array to the model via the ``__call__`` method.
        Arguments:
            input_array: The input that goes into the model. If it is a tuple, it gets
                interpreted as the sequence of positional arguments and is passed in by tuple unpacking.
                If it is a dict, the contents get passed in as named parameters by unpacking the dict.
                Otherwise, the input array gets passed in as a single argument.
        Returns:
            The output of the model.
        """
        return self.model(input_array, 0)


class VerificationCallbackBase(Callback):
    """
    Base class for model verification in form of a callback.
    This type of verification is expected to only work with
    :class:`~pytorch_lightning.core.lightning.LightningModule` and will take the input array
    from :attr:`~pytorch_lightning.core.lightning.LightningModule.example_input_array` if needed.
    """

    def __init__(self, warn: bool = True, error: bool = False):
        """
        Arguments:
            warn: If `True`, prints a warning message when verification fails. Default: `True`.
            error: If `True`, prints a error message when verification fails. Default: `False`.
        """
        self._raise_warning = warn
        self._raise_error = error

    def message(self, *args, **kwargs) -> str:
        """
        The message to be printed when the model does not pass the verification.
        If the message for warning and error differ, override the
        :meth:`VerificationCallbackBase.warning_message` and :meth:`VerificationCallbackBase.error_message`
        methods directly.
        Arguments:
            *args: Any positional arguments that are needed to construct the message.
            **kwargs: Any keyword arguments that are needed to construct the message.
        Returns:
            The message as a string.
        """
        pass

    def warning_message(self, *args, **kwargs) -> str:
        """ The warning message printed when the model does not pass the verification. """
        return self.message(*args, **kwargs)

    def error_message(self, *args, **kwargs) -> str:
        """ The error message printed when the model does not pass the verification. """
        return self.message(*args, **kwargs)

    def _raise(self, *args, **kwargs):
        if self._raise_error:
            raise RuntimeError(self.error_message(*args, **kwargs))
        if self._raise_warning:
            rank_zero_warn(self.warning_message(*args, **kwargs))


class BatchNormVerification(VerificationBase):

    normalization_layer = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._hook_handles = []
        self._module_sequence = []
        self._detected_pairs = []

    @property
    def detected_pairs(self) -> List[Tuple]:
        return self._detected_pairs

    def check(self, input_array=None) -> bool:
        input_array = self._get_input_array_copy(input_array)
        self.register_hooks()
        # trigger the hooks and collect sequence of layers
        self._model_forward(input_array)
        self.destroy_hooks()
        self.collect_detections()
        return not self._detected_pairs

    def collect_detections(self):
        detected_pairs = []
        for (name0, mod0), (name1, mod1) in zip(
            self._module_sequence[:-1], self._module_sequence[1:]
        ):
            bias = getattr(mod0, "bias", None)
            detected = (
                isinstance(mod1, self.normalization_layer)
                and mod1.training  # TODO: do we want/need this check?
                and isinstance(bias, torch.Tensor)
                and bias.requires_grad
            )
            if detected:
                detected_pairs.append((name0, name1))
        self._detected_pairs = detected_pairs
        return detected_pairs

    def register_hooks(self):
        hook_handles = []
        for name, module in self.model.named_modules():
            handle = module.register_forward_hook(self._create_hook(name))
            hook_handles.append(handle)
        self._hook_handles = hook_handles

    def _create_hook(self, module_name) -> Callable:
        def hook(module, inp_, out_):
            self._module_sequence.append((module_name, module))

        return hook

    def destroy_hooks(self):
        for hook in self._hook_handles:
            hook.remove()
        self._hook_handles = []


class BatchNormVerificationCallback(VerificationCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._verification = None

    def message(self, detections: List[Tuple]) -> str:
        first_detection = detections[0]
        message = (
            f"Detected a layer '{first_detection[0]}' with bias followed by"
            f" a normalization layer '{first_detection[1]}'."
            f" This makes the normalization ineffective and can lead to unstable training."
            f" Either remove the normalization or turn off the bias."
        )
        return message

    def on_train_start(self, trainer, pl_module):
        self._verification = BatchNormVerification(pl_module)
        self._verification.register_hooks()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx > 0:
            return
        detected_pairs = self._verification.collect_detections()
        if detected_pairs:
            self._raise(detections=detected_pairs)
        self._verification.destroy_hooks()


class BatchGradientVerification(VerificationBase):
    """
    Checks if a model mixes data across the batch dimension.
    This can happen if reshape- and/or permutation operations are carried out in the wrong order or
    on the wrong tensor dimensions.
    """

    def check(
        self,
        input_array=None,
        input_mapping: Callable = None,
        output_mapping: Callable = None,
        sample_idx=0,
    ) -> bool:
        """
        Runs the test for data mixing across the batch.
        Arguments:
            input_array: A dummy input for the model. Can be a tuple or dict in case the model takes
                multiple positional or named arguments.
            input_mapping: An optional input mapping that returns all batched tensors in a input collection.
                By default, we handle nested collections (tuples, lists, dicts) of tensors and pull them
                out. If your batch is a custom object, you need to provide this input mapping yourself.
                See :func:`default_input_mapping` for more information on the default behavior.
            output_mapping: An optional output mapping that combines all batched tensors in the output
                collection into one big batch of shape (B, N), where N is the total number of dimensions
                that follow the batch dimension in each tensor. By default, we handle nested collections
                (tuples, lists, dicts) of tensors and combine them automatically. See
                :func:`default_output_mapping` for more information on the default behavior.
            sample_idx:
                The index `i` of the batch sample to run the test for. When computing the gradient of
                a loss value on the `i-th` output w.r.t. the whole input, we expect the gradient to be
                non-zero only on the `i-th` input sample and zero gradient on the rest of the batch.
                Default: `i = 0`.
        Returns:
            `True` if the data in the batch does not mix during the forward pass, and `False` otherwise.
        """
        input_mapping = input_mapping or default_input_mapping
        output_mapping = output_mapping or default_output_mapping
        input_array = self._get_input_array_copy(input_array)
        input_batches = input_mapping(input_array)

        if input_batches[0].size(0) < 2:
            raise MisconfigurationException(
                "Batch size must be greater than 1 to run verification."
            )

        input_batches[0].requires_grad = True
        # for input_batch in input_batches:
        #     input_batch.requires_grad = True

        self.model.zero_grad()
        output = self._model_forward(input_array)

        # backward on the i-th sample should lead to gradient only in i-th input slice
        output_mapping(output).backward()

        zero_grad_inds = list(range(len(input_batches[0])))
        zero_grad_inds.pop(sample_idx)

        has_grad_outside_sample = [input_batches[0].grad[zero_grad_inds].abs().sum().item()]
        has_grad_inside_sample = [input_batches[0].grad[sample_idx].abs().sum().item()]
        return not any(has_grad_outside_sample) and all(has_grad_inside_sample)


class BatchGradientVerificationCallback(VerificationCallbackBase):
    """
    The callback version of the :class:`BatchGradientVerification` test.
    Verification is performed right before training begins.
    """

    def __init__(
        self,
        input_mapping: Callable = None,
        output_mapping: Callable = None,
        sample_idx=0,
        **kwargs
    ):
        """
        Arguments:
            input_mapping: An optional input mapping that returns all batched tensors in a input collection.
                See :meth:`BatchGradientVerification.check` for more information.
            output_mapping: An optional output mapping that combines all batched tensors in the output
                collection into one big batch. See :meth:`BatchGradientVerification.check` for more information.
            sample_idx: The index of the batch sample to run the test for.
                See :meth:`BatchGradientVerification.check` for more information.
            **kwargs: Additional arguments for the base class :class:`VerificationCallbackBase`
        """
        super().__init__(**kwargs)
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._sample_idx = sample_idx

    def message(self):
        message = (
            "Your model is mixing data across the batch dimension."
            " This can lead to wrong gradient updates in the optimizer."
            " Check the operations that reshape and permute tensor dimensions in your model."
        )
        return message

    def on_train_start(self, trainer, pl_module):
        verification = BatchGradientVerification(pl_module)
        if pl_module.example_input_array is None:
            pl_module.example_input_array = next(iter(trainer.datamodule.train_dataloader()))
        result = verification.check(
            input_array=pl_module.example_input_array,
            input_mapping=self._input_mapping,
            output_mapping=self._output_mapping,
            sample_idx=self._sample_idx,
        )
        if not result:
            self._raise()


def default_input_mapping(data: Any) -> List[torch.Tensor]:
    """
    Finds all tensors in a (nested) collection that have the same batch size.
    Args:
        data: a tensor or a collection of tensors (tuple, list, dict, etc.).
    Returns:
        A list of all tensors with the same batch dimensions. If the input was already a tensor, a one-
        element list with the tensor is returned.
    >>> data = (torch.zeros(3, 1), "foo", torch.ones(3, 2), torch.rand(2))
    >>> result = default_input_mapping(data)
    >>> len(result)
    2
    >>> result[0].shape
    torch.Size([3, 1])
    >>> result[1].shape
    torch.Size([3, 2])
    """
    tensors = collect_tensors(data)
    batches = []
    for tensor in tensors:
        if tensor.ndim > 0 and (not batches or tensor.size(0) == batches[0].size(0)):
            batches.append(tensor)
    return batches


def default_output_mapping(data: Any) -> torch.Tensor:
    """
    Pulls out all tensors in a output collection and combines them into one big batch
    for verification.
    Args:
        data: a tensor or a (nested) collection of tensors (tuple, list, dict, etc.).
    Returns:
        A float tensor with shape (B, N) where B is the batch size and N is the sum of (flattened)
        dimensions of all tensors in the collection. If the input was already a tensor, the tensor
        itself is returned.
    Example:
        >>> data = (torch.rand(3, 5), "foo", torch.rand(3, 2, 4))
        >>> result = default_output_mapping(data)
        >>> result.shape
        torch.Size([3, 13])
        >>> data = {"one": torch.rand(3, 5), "two": torch.rand(3, 2, 1)}
        >>> result = default_output_mapping(data)
        >>> result.shape
        torch.Size([3, 7])
    """
    if isinstance(data, torch.Tensor):
        return data

    batches = default_input_mapping(data)
    # cannot use .flatten(1) because of tensors with shape (B, )
    batches = [batch.view(batch.size(0), -1).float() for batch in batches]
    combined = torch.cat(batches, 1)  # combined batch has shape (B, N)
    return combined


def collect_tensors(data: Any) -> List[torch.Tensor]:
    """ Filters all tensors in a collection and returns them in a list. """
    tensors = []

    def collect_batches(tensor):
        tensors.append(tensor)
        return tensor

    apply_to_collection(data, dtype=torch.Tensor, function=collect_batches)
    return tensors
