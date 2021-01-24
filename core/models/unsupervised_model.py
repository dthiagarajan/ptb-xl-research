import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.distributed as dist  # noqa: F401

from core.callbacks.model_callbacks import callback_factory
from core.distributed.ops import all_gather_op
import core.models as ptbxl_models
from core.models import CallbackModel, Flatten


class PTBXLUnsupervisedModel(LightningModule):
    """Lightning module wrapper for training a generic model for PTB-XL self-supervision"""

    allowed_models = [
        'Simple1DCNN',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'wide_resnet18_2', 'wide_resnet34_2', 'wide_resnet50_2', 'wide_resnet101_2',
        'wide_resnet18_4', 'wide_resnet34_4', 'wide_resnet50_4', 'wide_resnet101_4',
        'wide_resnet18_8', 'wide_resnet34_8', 'wide_resnet50_8', 'wide_resnet101_8',
    ]

    def __init__(self, model_name, *args, **kwargs):
        super(PTBXLUnsupervisedModel, self).__init__()
        assert model_name in self.allowed_models, f"Please pick one of: {self.allowed_models}"
        self.model_name = model_name
        self.maybe_set(kwargs, 'task_name')
        self.maybe_set(kwargs, 'logger_platform')
        self.maybe_set(kwargs, 'show_heatmaps')
        self.maybe_set(kwargs, 'data_dir')
        self.maybe_set(kwargs, 'batch_size')
        self.maybe_set(kwargs, 'num_input_channels')
        self.maybe_set(kwargs, 'num_classes')
        self.maybe_set(kwargs, 'num_workers')
        self.maybe_set(kwargs, 'lr', mod=sorted)
        if len(self.lr) == 1:
            self.lr = self.lr[0]
        self.maybe_set(kwargs, 'use_one_cycle_lr_scheduler')
        self.maybe_set(kwargs, 'lr_decay')
        self.maybe_set(kwargs, 'lr_decay_period')
        self.maybe_set(kwargs, 'lr_decay_gamma')
        self.maybe_set(kwargs, 'max_epochs')
        self.model = self.initialize_model(model_name, **kwargs)
        self.maybe_set(kwargs, 'sampling_rate')
        self.maybe_set(kwargs, 'profiler')
        self.maybe_set(kwargs, 'model_checkpoint')
        self.maybe_set(kwargs, 'model_name')
        kwargs['model_name'] = model_name
        self.save_hyperparameters(*[k for k in kwargs.keys() if 'mapping' not in k])

        self.training_log_step, self.validation_log_step, self.testing_log_step = 0, 0, 0
        self.training_log_epoch, self.validation_log_epoch, self.testing_log_epoch = 0, 0, 0
        self.best_metrics = None

    def maybe_set(self, kwargs, attr, mod=lambda attr: attr):
        if attr in kwargs:
            setattr(self, attr, mod(kwargs[attr]))

    def initialize_model(self, model_name, **kwargs):
        # Arbitrarily set num_classes to 1
        model = getattr(ptbxl_models, model_name)(num_input_channels=kwargs['num_input_channels'], num_classes=1)
        layers = list(model.children())
        model = torch.nn.Sequential(*layers[:-1])
        projection_head = torch.nn.Linear(layers[-1].in_channels, kwargs['projection_dim'])
        return CallbackModel(
            torch.nn.Sequential(model, Flatten(), projection_head),
            callbacks=callback_factory(unsupervised=True, **kwargs),
            profile=kwargs['profiler'],
            unsupervised=True
        )

    def log_hyperparams(self):
        if self.logger_platform == 'tensorboard':
            try:
                self.logger.log_hyperparams(self.hparams, self.best_metrics)
            except TypeError:
                print(f'Using a logger that does not log metrics with hyperparams.')
        elif self.logger_platform == 'wandb':
            try:
                self.logger.log_hyperparams(self.hparams)
                self.logger.experiment.summary.update(self.best_metrics)
            except AttributeError:
                print(
                    f'Logger experiment is mocked for LR find - skipping hyperparameter logging.'
                )

    def update_hyperparams_and_metrics(self, metrics):
        if self.best_metrics is None:
            self.best_metrics = {f'best_{k}': v for (k, v) in metrics.items()}
            self.best_metrics['best_epoch'] = 0
        else:
            flag = True
            for k in metrics:
                if 'loss' in k:
                    if metrics[k] >= self.best_metrics[f'best_{k}']:
                        flag = False
                        break
                elif 'acc' in k or 'auc' in k:
                    if metrics[k] <= self.best_metrics[f'best_{k}']:
                        flag = False
                        break
            if flag is True:
                self.best_metrics = {f'best_{k}': v for (k, v) in metrics.items()}
                self.best_metrics['best_epoch'] = self.validation_log_epoch
        self.log_hyperparams()

    def setup(self, stage):
        # Need this function to have best metrics being logged in hyperparameters tab of TB
        self.logger.testing = (stage == 'test')
        self.update_hyperparams_and_metrics(
            {
                'val_epoch_loss': float('inf'),
                'val_epoch_acc': 0,
                'val_epoch_auc': 0,
                'val_epoch_f_max': 0
            }
        )

    def forward(self, batch, batch_idx):
        return self.model.run(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        output_metrics = self(batch, batch_idx)
        return output_metrics

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        output_metrics = outputs[0][0]['extra']
        output_metrics['loss'] = outputs[0][0]['minimize']
        output_metrics = self.all_gather_outputs([output_metrics])
        tensorboard_logs = {
            f'Train/train_step_{metric_name}': output_metrics[metric_name]
            for metric_name in output_metrics if metric_name not in ['similarities']
        }
        self.logger.log_metrics(tensorboard_logs, self.training_log_step)
        self.training_log_step += 1

    def training_epoch_end(self, outputs):
        loss_mean, prob_mean, similarities = self.all_gather_outputs(outputs, detach=True).values()
        self.logger.log_metrics(
            {
                "Train/train_loss": loss_mean,
                "Train/train_prob": prob_mean,
            },
            self.training_log_epoch + 1
        )
        try:
            self.logger.add_scalars(
                "Losses", {"train_loss": loss_mean}, self.training_log_epoch + 1
            )
            self.logger.add_scalars(
                "Probabilities", {"train_prob": prob_mean}, self.training_log_epoch + 1
            )
        except AttributeError as e:
            print(f'In (train) LR find, error ignored: {str(e)}')
        if self.profiler and len(self.model.timings) > 0:
            hook_reports = []
            for hook_name in self.model.timings:
                for callback_name, times in self.model.timings[hook_name].items():
                    times = np.array(times)
                    mean_time, sum_time = times.mean(), times.sum()
                    hook_reports.append({
                        'Callback Name': callback_name,
                        'Mean time (s)': mean_time,
                        'Sum time (s)': sum_time
                    })
            print(pd.DataFrame(hook_reports))
            hook_reports = pd.DataFrame(hook_reports).set_index('Callback Name')
            print(hook_reports)
            self.model.timings = {}

        self.training_log_epoch += 1

    def validation_step(self, batch, batch_idx):
        output_metrics = self(batch, batch_idx)
        return output_metrics

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        output_metrics = outputs
        output_metrics = self.all_gather_outputs([output_metrics])
        tensorboard_logs = {
            f'Validation/val_step_{metric_name}': output_metrics[metric_name]
            for metric_name in output_metrics if metric_name not in ['similarities']
        }
        self.logger.log_metrics(tensorboard_logs, self.validation_log_step)
        self.validation_log_step += 1

    def validation_epoch_end(self, outputs):
        val_loss_mean, val_prob_mean, val_similarities = self.all_gather_outputs(outputs, detach=True).values()

        self.logger.log_metrics(
            {
                "Validation/val_loss": val_loss_mean,
                "Validation/val_prob": val_prob_mean
            },
            self.validation_log_epoch + 1
        )
        try:
            self.logger.add_scalars(
                "Losses", {"val_loss": val_loss_mean}, self.validation_log_epoch + 1
            )
            self.logger.add_scalars(
                "Probabilities", {"val_prob": val_prob_mean}, self.validation_log_epoch + 1
            )
        except AttributeError as e:
            print(f'In (validation) LR find, error ignored: {str(e)}')
        self.update_hyperparams_and_metrics({'val_epoch_loss': val_loss_mean, 'val_epoch_prob': val_prob_mean})
        if self.profiler and len(self.model.timings) > 0:
            hook_reports = []
            for hook_name in self.model.timings:
                for callback_name, times in self.model.timings[hook_name].items():
                    times = np.array(times)
                    mean_time, sum_time = times.mean(), times.sum()
                    hook_reports.append({
                        'Callback Name': callback_name,
                        'Mean time (s)': mean_time,
                        'Sum time (s)': sum_time
                    })
            hook_reports = pd.DataFrame(hook_reports).set_index('Callback Name')
            print(hook_reports)
            self.model.timings = {}
        self.validation_log_epoch += 1
        return {
            'val_epoch_loss': val_loss_mean,
            'val_epoch_prob': val_prob_mean
        }

    def all_gather_outputs(self, outputs, detach=False):
        losses = torch.stack([x['loss'] for x in outputs])
        probs = torch.stack([x['correct_similarity_prob'] for x in outputs])
        similarities = torch.cat([x['similarities'] for x in outputs])
        if detach:
            probs, similarities = probs.detach(), similarities.detach()

        if 'CPU' in self.trainer.accelerator_backend.__class__.__name__:
            return {
                'loss': losses.mean(),
                'prob': probs.mean(),
                'similarities': similarities,
            }

        return {
            'loss': all_gather_op(losses).mean(),
            'prob': all_gather_op(probs).mean(),
            'similarities': all_gather_op(similarities)
        }

    @property
    def version_directory(self):
        if not hasattr(self, '_version_directory'):
            version_cnt = 0
            filedir = Path(self.log_dir, f'version_{version_cnt}')
            while Path(self.log_dir, f'version_{version_cnt}').is_dir():
                filedir = Path(self.log_dir, f'version_{version_cnt}')
                version_cnt += 1
            self._version_directory = filedir
        return self._version_directory

    def checkpoint_object(self, obj, filename):
        filedir = self.version_directory
        filepath = os.path.join(filedir, f'{filename}.pkl')
        print(f'Dumping object to {filepath}...')
        Path(filepath).touch()
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f'...done.')
        print(f"To load, run the following in Python: pickle.load(open('{filepath}', 'rb'))")

    def configure_optimizers(self):
        if type(self.lr) is float:
            optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.lr}])
        else:
            param_lr_mappings, self.lr = self.get_param_lr_maps(self.lr)
            optimizer = torch.optim.AdamW(param_lr_mappings)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='Simple1DCNN')
        parser.add_argument(
            '--logger_platform', type=str, choices=['tensorboard', 'wandb'], default='tensorboard'
        )
        parser.add_argument('--num_input_channels', type=int, default=12)
        parser.add_argument('--projection_dim', type=int, default=128)
        parser.add_argument('--lr', nargs='+', type=float, default=[1e-3])
        parser.add_argument('--temperature', type=float, default=0.5)
        return parser
