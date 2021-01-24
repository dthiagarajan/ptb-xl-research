import time
import torch.nn as nn
from typing import Callable, List


class CallbackModel(nn.Module):
    def __init__(self, model: nn.Module, callbacks: List[Callable], profile=False, unsupervised=False):
        super(CallbackModel, self).__init__()
        self.model = model
        self._callbacks = callbacks
        self.profile = profile
        self.timings = {}
        self.unsupervised = unsupervised

    def callback(self, hook_name, batch, batch_idx, model_output=None):
        if hook_name not in self.timings:
            self.timings[hook_name] = {}

        for callback in self._callbacks:
            hook_fx = getattr(callback, hook_name)
            start_time = time.time()
            model_output = hook_fx(
                self, batch, batch_idx, model_output=model_output
            )
            if callback.__class__.__name__ not in self.timings[hook_name]:
                self.timings[hook_name][callback.__class__.__name__] = []
            self.timings[hook_name][callback.__class__.__name__].append(time.time() - start_time)
        return model_output

    def run(self, batch, batch_idx):
        self.next_batch = batch
        # These callbacks might modify self.next_batch
        self.callback('on_forward_start', batch, batch_idx)
        try:
            x, _ = self.next_batch
        except ValueError:
            assert self.unsupervised, 'Model is assumed to get batches with no labels.'
            x = self.next_batch
        model_output = self.model(x)
        modified_model_output = self.callback(
            'on_forward_end', batch, batch_idx, model_output=model_output
        )
        return modified_model_output
