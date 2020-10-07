import functools
import numpy as np
import os
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
import torch
from typing import Dict, List, Optional


class MultiMetricModelCheckpoint(ModelCheckpoint):
    r"""
    Save the model after every epoch if it improves on multiple metrics. Documentation taken largely
    from the ModelCheckpoint in PyTorch Lightning.

    After training finishes, use :attr:`best_model_path` to retrieve the path to the
    best checkpoint file and :attr:`best_model_score` to retrieve its score.

    Args:
        filepath: path to save the model file.
            Can contain named formatting options to be auto-filled.

            Example::

                # custom path
                # saves a file like: my/path/epoch_0.ckpt
                >>> checkpoint_callback = MultiMetricModelCheckpoint('my/path/')

                # save any arbitrary metrics like `val_loss`, etc. in name
                # saves a file like: my/path/epoch=2-val_loss=0.2_other_metric=0.3.ckpt
                >>> checkpoint_callback = MultiMetricModelCheckpoint(
                ...     filepath='my/path/{epoch}-{val_loss:.2f}-{other_metric:.2f}'
                ... )

            Can also be set to `None`, then it will be set to default location
            during trainer construction.

        monitors: quantities to monitor.
        verbose: verbosity mode. Default: ``False``.
        save_last: always saves the model at the end of the epoch. Default: ``False``.
        save_top_k: if `save_top_k == k`,
            the best k models according to
            the quantities monitored will be saved.
            if ``save_top_k == 0``, no models are saved.
            if ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if ``save_top_k >= 2`` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        modes: one of {min, max} for each monitor.
            If ``save_top_k != 0``, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc.
        save_weights_only: if ``True``, then only the model's weights will be
            saved (``model.save_weights(filepath)``), else the full model
            is saved (``model.save(filepath)``).
        period: Interval (number of epochs) between checkpoints.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from core.callbacks import MultiMetricModelCheckpoint

        # saves checkpoints to 'my/path/' whenever 'val_loss' has a new min
        >>> checkpoint_callback = MultiMetricModelCheckpoint(filepath='my/path/')
        >>> trainer = Trainer(checkpoint_callback=checkpoint_callback)

        # save epoch and val_loss in name
        # saves a file like: my/path/sample-mnist_epoch=02_val_loss=0.32.ckpt
        >>> checkpoint_callback = MultiMetricModelCheckpoint(
        ...     filepath='my/path/sample-mnist_{epoch:02d}-{val_loss:.2f}'
        ... )

        # retrieve the best checkpoint after training
        checkpoint_callback = MultiMetricModelCheckpoint(filepath='my/path/')
        trainer = Trainer(checkpoint_callback=checkpoint_callback)
        model = ...
        trainer.fit(model)
        checkpoint_callback.best_model_path

    """
    def __init__(self, filepath: Optional[str] = None, monitors: List[str] = ['val_loss'],
                 verbose: bool = False, save_last: bool = False, save_weights_only: bool = False,
                 modes: List[str] = ['min'], period: int = 1, prefix: str = ''):
        super().__init__()
        if filepath is not None and os.path.isdir(filepath) and len(os.listdir(filepath)) > 0:
            rank_zero_warn(
                f"Checkpoint directory {filepath} exists and is not empty."
                "All files in this directory will be deleted when a checkpoint is saved!"
            )
        self._rank = 0

        self.monitors = monitors
        self.verbose = verbose
        if filepath is None:  # will be determined by trainer at runtime
            self.dirpath, self.filename = None, None
        else:
            if os.path.isdir(filepath):
                self.dirpath, self.filename = filepath, '{epoch}'
            else:
                filepath = os.path.realpath(filepath)
                self.dirpath, self.filename = os.path.split(filepath)
            os.makedirs(self.dirpath, exist_ok=True)
        self.save_last = save_last
        self.save_top_k = 1
        self.save_weights_only = save_weights_only
        self.period = period
        self.epoch_last_check = None
        self.prefix = prefix
        self.best_k_models = {}
        # {filename: monitor}
        self.kth_best_model_path = ''
        self.best_model_score = 0
        self.best_model_path = ''
        self.save_function = None

        torch_inf = torch.tensor(np.Inf)
        mode_dict = {
            'min': (torch_inf, 'min'),
            'max': (-torch_inf, 'max'),
        }

        for mode in modes:
            assert mode in mode_dict, \
                f'MultiMetricModelCheckpoint mode {mode} is unknown, use 1 of [min, max].'

        self.kth_values = {
            monitor: mode_dict[mode][0] for (monitor, mode) in zip(monitors, modes)
        }
        self.modes = modes
        # self.kth_value, self.mode = mode_dict[mode]

    def check_monitor_top_k(self, current: Dict):
        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        for k in self.monitors:
            if not isinstance(current[k], torch.Tensor):
                rank_zero_warn(
                    f'{current[k]} is supposed to be a `torch.Tensor`. Saving checkpoint may not work correctly.'
                    f' HINT: check the value of {k} in your validation loop', RuntimeWarning
                )
                current[k] = torch.tensor(current[k])

        monitor_op = {
            "min": torch.lt,
            "max": torch.gt,
        }

        return functools.reduce(
            np.logical_and,
            [
                monitor_op[mode](
                    current[k].cpu(), self.best_k_models[self.kth_best_model_path][k].cpu()
                )
                for k, mode in zip(self.monitors, self.modes)
            ]
        )

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # only run on main process
        if trainer.global_rank != 0:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epoch_last_check is not None and (epoch - self.epoch_last_check) < self.period:
            # skipping in this term
            return

        self.epoch_last_check = epoch

        if self.save_last:
            filepath = os.path.join(self.dirpath, self.prefix + 'last.ckpt')
            self._save_model(filepath)

        filepath = self.format_checkpoint_name(epoch, metrics)
        version_cnt = 0
        while os.path.isfile(filepath):
            filepath = self.format_checkpoint_name(epoch, metrics, ver=version_cnt)
            # this epoch called before
            version_cnt += 1

        if self.save_top_k != -1:
            current = {monitor: metrics.get(monitor) for monitor in self.monitors}
            none_flag = False
            for monitor in self.monitors:
                if not isinstance(current[monitor], torch.Tensor):
                    rank_zero_warn(
                        f'The metric you returned {current} must be a `torch.Tensor` instance, checkpoint not saved'
                        f' HINT: what is the value of {monitor} in validation_epoch_end()?', RuntimeWarning
                    )
                    if current[monitor] is not None:
                        current[monitor] = torch.tensor(current[monitor])
                    else:
                        none_flag = True
                        break

            if none_flag:
                rank_zero_warn(
                    f'Can save best model only with {self.monitors} available, skipping.', RuntimeWarning
                )
            elif self.check_monitor_top_k(current):
                self._do_check_save(filepath, current, epoch, trainer, pl_module)
            elif self.verbose > 0:
                log.info(f'\nEpoch {epoch:05d}: {self.monitors}  was not in top {self.save_top_k}')

        else:
            if self.verbose > 0:
                log.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')

            assert trainer.global_rank == 0, 'tried to make a checkpoint from non global_rank=0'
            self._save_model(filepath)

    def _do_check_save(self, filepath, current, epoch, trainer, pl_module):
        # remove kth
        del_list = []
        if len(self.best_k_models) == self.save_top_k and self.save_top_k > 0:
            delpath = self.kth_best_model_path
            self.best_k_models.pop(self.kth_best_model_path)
            del_list.append(delpath)

        self.best_k_models[filepath] = current
        if len(self.best_k_models) == self.save_top_k:
            # monitor dict has reached k elements
            self.kth_best_model_path = filepath
            self.kth_value = self.best_k_models[filepath]

        self.best_model_path = filepath
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose > 0:
            log.info(
                f'\nEpoch {epoch:05d}: {self.monitors} reached'
                f' {current} (best {self.best_model_score}), saving model to'
                f' {filepath} as top {self.save_top_k}')
        self._save_model(filepath, trainer, pl_module)

        for cur_path in del_list:
            if cur_path != filepath:
                self._del_model(cur_path)
