import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.distributed as dist  # noqa: F401
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from core.analysis.visualization import compute_and_show_heatmap
from core.callbacks.model_callbacks import callback_factory
from core.distributed.ops import all_gather_op
from core.metrics import metric_summary
import core.models as ptbxl_models
from core.models import CallbackModel


class PTBXLClassificationModel(LightningModule):
    """Lightning module wrapper for training a generic model for PTB-XL classification"""

    allowed_models = [
        'Simple1DCNN',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'wide_resnet18_2', 'wide_resnet34_2', 'wide_resnet50_2', 'wide_resnet101_2',
        'wide_resnet18_4', 'wide_resnet34_4', 'wide_resnet50_4', 'wide_resnet101_4',
        'wide_resnet18_8', 'wide_resnet34_8', 'wide_resnet50_8', 'wide_resnet101_8',
    ]

    def __init__(self, model_name, *args, **kwargs):
        super(PTBXLClassificationModel, self).__init__()
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
        self.maybe_set(kwargs, 'class_weighted_loss')
        self.model = self.initialize_model(model_name, **kwargs)
        self.maybe_set(kwargs, 'label_counts_mapping', mod=str)
        self.maybe_set(kwargs, 'label_weight_mapping', mod=str)
        self.maybe_set(kwargs, 'sampling_rate')
        self.maybe_set(kwargs, 'profiler')
        self.maybe_set(kwargs, 'heatmap_layers')
        self.maybe_set(kwargs, 'model_checkpoint')
        self.maybe_set(kwargs, 'model_name')
        self.maybe_set(kwargs, 'swa')
        self.maybe_set(kwargs, 'swa_epochs')
        self.maybe_set(kwargs, 'swa_lr')
        kwargs['model_name'] = model_name
        self.save_hyperparameters(*[k for k in kwargs.keys() if 'mapping' not in k])

        self.training_log_step, self.validation_log_step, self.testing_log_step = 0, 0, 0
        self.training_log_epoch, self.validation_log_epoch, self.testing_log_epoch = 0, 0, 0
        self.best_metrics = None

    def maybe_set(self, kwargs, attr, mod=lambda attr: attr):
        if attr in kwargs:
            setattr(self, attr, mod(kwargs[attr]))

    def initialize_model(self, model_name, **kwargs):
        model = getattr(ptbxl_models, model_name)(
            num_input_channels=kwargs['num_input_channels'], num_classes=kwargs['num_classes']
        )
        return CallbackModel(
            model, callbacks=callback_factory(**kwargs), profile=kwargs['profiler']
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
            for metric_name in output_metrics if metric_name not in ['probs', 'targets']
        }
        self.logger.log_metrics(tensorboard_logs, self.training_log_step)
        self.training_log_step += 1

    def all_gather_outputs(self, outputs, detach=False):
        losses = torch.stack([x['loss'] for x in outputs])
        accs = torch.stack([x['acc'] for x in outputs])
        aucs = torch.stack([x['auc'] for x in outputs])
        probs = torch.cat([x['probs'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        if detach:
            probs, targets = probs.detach(), targets.detach()

        if 'CPU' in self.trainer.accelerator_backend.__class__.__name__:
            return {
                'loss': losses.mean(),
                'acc': accs.mean(),
                'auc': aucs.mean(),
                'probs': probs,
                'targets': targets
            }

        return {
            'loss': all_gather_op(losses).mean(),
            'acc': all_gather_op(accs).mean(),
            'auc': all_gather_op(aucs).mean(),
            'probs': all_gather_op(probs),
            'targets': all_gather_op(targets)
        }

    def training_epoch_end(self, outputs):
        loss_mean, acc_mean, auc_mean, probs, targets = self.all_gather_outputs(outputs, detach=True).values()
        (
            f_max,
            _,
            f_scores,
            average_precisions,
            average_recalls,
            thresholds
        ) = metric_summary(targets.cpu().numpy(), probs.cpu().numpy())

        self.logger.log_metrics(
            {
                "Train/train_loss": loss_mean,
                "Train/train_acc": acc_mean,
                "Train/train_auc": auc_mean
            },
            self.training_log_epoch + 1
        )
        try:
            self.logger.add_scalars(
                "Losses", {"train_loss": loss_mean}, self.training_log_epoch + 1
            )
            self.logger.add_scalars(
                "Accuracies", {"train_acc": acc_mean}, self.training_log_epoch + 1
            )
            self.logger.add_scalars(
                "AUCs", {"train_auc": auc_mean}, self.training_log_epoch + 1
            )
            self.logger.add_scalars(
                "F1 Max Scores", {"train_f1-max": f_max}, self.training_log_epoch + 1
            )
        except AttributeError as e:
            print(f'In (train) LR find, error ignored: {str(e)}')
        metric_appendix = {}
        for f1_score, average_precision, average_recall, threshold in zip(
            f_scores, average_precisions, average_recalls, thresholds
        ):
            metric_appendix.update({
                f'Train Metric Appendix/F1-Score ({threshold:0.1f})': f1_score,
                f'Train Metric Appendix/Average Precision ({threshold:0.1f})': average_precision,
                f'Train Metric Appendix/Average Recall ({threshold:0.1f})': average_recall,
            })
        self.logger.log_metrics(metric_appendix, self.training_log_epoch + 1)
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

        if self.swa:
            if not hasattr(self, 'swa_model'):
                print(f'Initializing SWA model...')
                self.swa_model = AveragedModel(self.model.model)
                print(f'...done.')
            self.swa_model.update_parameters(self.model.model)
            torch.optim.swa_utils.update_bn(
                self.trainer.datamodule.train_dataloader(), self.swa_model, device=self.device
            )
        self.training_log_epoch += 1

    def finalize_swa_model(self):
        assert self.swa, 'Model must have been trained with SWA.'
        self.model.model = self.swa_model

    def validation_step(self, batch, batch_idx):
        output_metrics = self(batch, batch_idx)
        return output_metrics

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        output_metrics = outputs
        output_metrics = self.all_gather_outputs([output_metrics])
        tensorboard_logs = {
            f'Validation/val_step_{metric_name}': output_metrics[metric_name]
            for metric_name in output_metrics if metric_name not in ['probs', 'targets']
        }
        self.logger.log_metrics(tensorboard_logs, self.validation_log_step)
        self.validation_log_step += 1

    def validation_epoch_end(self, outputs):
        val_loss_mean, val_acc_mean, val_auc_mean, probs, targets = self.all_gather_outputs(outputs, detach=True).values()
        (
            f_max,
            _,
            f_scores,
            average_precisions,
            average_recalls,
            thresholds
        ) = metric_summary(targets.cpu().numpy(), probs.cpu().numpy())

        self.logger.log_metrics(
            {
                "Validation/val_loss": val_loss_mean,
                "Validation/val_acc": val_acc_mean,
                "Validation/val_auc": val_auc_mean,
            },
            self.validation_log_epoch + 1
        )
        try:
            self.logger.add_scalars(
                "Losses", {"val_loss": val_loss_mean}, self.validation_log_epoch + 1
            )
            self.logger.add_scalars(
                "Accuracies", {"val_acc": val_acc_mean}, self.validation_log_epoch + 1
            )
            self.logger.add_scalars(
                "AUCs", {"val_auc": val_auc_mean}, self.validation_log_epoch + 1
            )
            self.logger.add_scalars(
                "F1 Max Scores", {"val_f1-max": f_max}, self.validation_log_epoch + 1
            )
        except AttributeError as e:
            print(f'In (validation) LR find, error ignored: {str(e)}')
        metric_appendix = {}
        for f1_score, average_precision, average_recall, threshold in zip(
            f_scores, average_precisions, average_recalls, thresholds
        ):
            metric_appendix.update({
                f'Validation Metric Appendix/F1-Score ({threshold:0.1f})': f1_score,
                f'Validation Metric Appendix/Average Precision ({threshold:0.1f})': average_precision,
                f'Validation Metric Appendix/Average Recall ({threshold:0.1f})': average_recall,
            })
        self.logger.log_metrics(metric_appendix, self.validation_log_epoch + 1)
        self.update_hyperparams_and_metrics(
            {
                'val_epoch_loss': val_loss_mean,
                'val_epoch_acc': val_acc_mean,
                'val_epoch_auc': val_auc_mean,
                'val_epoch_f_max': torch.tensor(f_max)
            }
        )
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
            'val_epoch_auc': val_auc_mean,
            'val_epoch_f_max': torch.tensor(f_max)
        }

    def test_step(self, batch, batch_idx):
        output_metrics = self(batch, batch_idx)
        return output_metrics

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        output_metrics = outputs
        output_metrics = self.all_gather_outputs([output_metrics])
        tensorboard_logs = {
            f'Test/test_step_{metric_name}': output_metrics[metric_name]
            for metric_name in output_metrics if metric_name not in ['probs', 'targets']
        }
        self.logger.log_metrics(tensorboard_logs, self.testing_log_step)
        self.testing_log_step += 1


    def test_epoch_end(self, outputs):
        if hasattr(self, 'model_checkpoint') and int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print(f'Symlinking model checkpoint to this runs version directory...')
            filedir = self.version_directory
            symlinked_model_path = Path(filedir, 'checkpoints', self.model_checkpoint.split('/')[-1])
            os.symlink(Path(self.model_checkpoint).resolve(), symlinked_model_path)
            hparams_path = Path(Path(self.model_checkpoint).parent.parent.resolve(), 'hparams.yaml')
            symlinked_hparams_path = Path(filedir, 'hparams.yaml')
            os.symlink(hparams_path, symlinked_hparams_path)
            print(
                f'...done. Symlinked to {self.model_checkpoint} to {symlinked_model_path} and '
                f'{hparams_path} to {symlinked_hparams_path}.'
            )
        test_loss_mean, test_acc_mean, test_auc_mean, probs, targets = self.all_gather_outputs(outputs, detach=True).values()
        self.test_outputs = {
            'loss': test_loss_mean,
            'acc': test_acc_mean,
            'auc': test_auc_mean,
            'probs': probs,
            'targets': targets
        }
        if not hasattr(self, 'labels'):
            self.labels = self.trainer.datamodule.labels  # Needed for confusion matrix labels
        if self.show_heatmaps:
            with torch.set_grad_enabled(True):
                self.visualize_best_and_worst_heatmaps(probs.numpy(), targets.numpy())
            if hasattr(self, 'model_checkpoint') and int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print(f'Deleting model checkpoint and symlink to save memory...')
                symlinked_model_path.unlink()
                os.remove(self.model_checkpoint)
                print(
                    f'...done. Removed link at {symlinked_model_path}, '
                    f'deleted {self.model_checkpoint}.'
                )
        f_max = metric_summary(targets.cpu().numpy(), probs.cpu().numpy())[0]
        self.logger.plot_confusion_matrix(
            targets.cpu().numpy(), (probs > 0.5).cpu().numpy(), self.labels
        )
        self.logger.plot_roc(targets.long().cpu().numpy(), probs.cpu().numpy(), self.labels)
        self.log_dict(
            {
                'test_epoch_loss': test_loss_mean,
                'test_epoch_acc': test_acc_mean,
                'test_epoch_auc': test_auc_mean,
                'test_epoch_f_max': f_max
            },
            on_step=False,
            on_epoch=True
        )

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

    def checkpoint_fig(self, fig, subdir, filename, verbose=False):
        filedir = self.version_directory
        filepath = Path(filedir, subdir)
        if verbose:
            print(f'Dumping figure to {filepath}...')
        filepath.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(filepath, filename))
        if verbose:
            print(f'...done.')

    def get_and_save_heatmap_figures(
        self,
        datapoints,
        label_mapping,
        label_name,
        datapoint_identifier,
        model_layer='pool3',
        index_subset=None
    ):
        figures_plotted = 0
        subdir = '_'.join(datapoint_identifier.lower().split('/'))
        indices = []
        _datapoints = datapoints
        if index_subset is not None:
            _datapoints = [datapoints[i] for i in index_subset]
        for i, datapoint in enumerate(tqdm(
            _datapoints,
            desc=f'{model_layer} {str(label_name)} {datapoint_identifier} heatmap generation'
        )):
            if figures_plotted > 5:
                break
            fig = compute_and_show_heatmap(
                self.model.model, model_layer,
                datapoint, label_mapping, class_of_concern=label_name,
                show_fig=False, figsize=(10, 40)
            )
            if figures_plotted <= 5 and fig is not None:
                indices.append(i)
                if not self.suppress_logging:
                    self.logger.plot_figures({
                        f'{label_name} {datapoint_identifier} (Rank {i})': fig
                    })
                figures_plotted += 1
                index = i
                if index_subset is not None:
                    index = index_subset[i]
                self.checkpoint_fig(
                    fig,
                    Path('heatmaps', label_name, model_layer, subdir),
                    f'{str(index)}.jpg'
                )
        return indices

    def visualize_best_and_worst_heatmaps(self, test_probs, test_targets):
        label_mapping = {k: i for i, k in enumerate(self.labels)}
        test_diffs = np.abs(test_probs - test_targets)
        indices_for_inspection = {}
        for label_name, label_index in label_mapping.items():
            test_label_targets = test_targets[:, label_index]

            test_label_diffs = test_diffs[:, label_index].copy()
            # Best positives, worst positives
            positive_mask = test_label_targets == 1
            test_label_diffs[~positive_mask] = 1.1
            all_best_present_indices = test_label_diffs.argsort()
            best_indices = all_best_present_indices[:50]
            test_label_diffs[~positive_mask] = -0.1
            test_label_diffs *= -1
            all_worst_present_indices = test_label_diffs.argsort()
            worst_indices = all_worst_present_indices[:50]
            best_test_positive_datapoints = [
                self.trainer.datamodule.val_dataset[i] for i in best_indices
            ]
            worst_test_positive_datapoints = [
                self.trainer.datamodule.val_dataset[i] for i in worst_indices
            ]

            test_label_diffs = test_diffs[:, label_index].copy()
            # Best negatives, worst negatives
            negative_mask = test_label_targets == 0
            test_label_diffs[~negative_mask] = 1.1
            all_best_absent_indices = test_label_diffs.argsort()
            best_indices = all_best_absent_indices[:50]
            test_label_diffs[~negative_mask] = -0.1
            test_label_diffs *= -1
            all_worst_absent_indices = test_label_diffs.argsort()
            worst_indices = all_worst_absent_indices[:50]

            indices_for_inspection[str(label_name)] = {
                'best_present': all_best_present_indices,
                'worst_present': all_worst_present_indices,
                'best_absent': all_best_absent_indices,
                'worst_absent': all_worst_absent_indices
            }

            best_test_negative_datapoints = [
                self.trainer.datamodule.val_dataset[i] for i in best_indices
            ]
            worst_test_negative_datapoints = [
                self.trainer.datamodule.val_dataset[i] for i in worst_indices
            ]
            # Assume that heatmap layers are given in sorted order, i.e. from beginning to end
            datapoint_indices = None
            for i, layer in enumerate(self.heatmap_layers[::-1]):
                if i == 0:
                    datapoint_indices = self.get_and_save_heatmap_figures(
                        best_test_positive_datapoints,
                        label_mapping,
                        label_name,
                        'Present/Best',
                        model_layer=layer
                    )
                else:
                    self.get_and_save_heatmap_figures(
                        best_test_positive_datapoints,
                        label_mapping,
                        label_name,
                        'Present/Best',
                        model_layer=layer,
                        index_subset=datapoint_indices
                    )
                # self.get_and_save_heatmap_figures(
                #     worst_test_positive_datapoints,
                #     label_mapping,
                #     label_name,
                #     'Present/Worst',
                #     model_layer=layer
                # )
                # self.get_and_save_heatmap_figures(
                #     best_test_negative_datapoints,
                #     label_mapping,
                #     label_name,
                #     'Absent/Best',
                #     model_layer=layer
                # )
                # self.get_and_save_heatmap_figures(
                #     worst_test_negative_datapoints,
                #     label_mapping,
                #     label_name,
                #     'Absent/Worst',
                #     model_layer=layer
                # )

        self.checkpoint_object(indices_for_inspection, 'best_and_worst_datapoints')

    def get_param_lr_maps(self, lrs):
        """ Output parameter LR mappings for setting up an optimizer for `model`."""
        body_parameters = [param for (_, param) in self.model.model.named_parameters()]
        param_lr_mappings = []
        incr_size = len(body_parameters) // (len(lrs) - 1)
        for i in range(0, len(body_parameters), incr_size):
            submodel_lrs = np.geomspace(
                lrs[i // incr_size], lrs[min((i // incr_size) + 1, len(lrs) - 1)],
                len(body_parameters[i:i + incr_size])
            )
            param_lr_mappings.extend([
                {'params': bp, 'lr': submodel_lr} for bp, submodel_lr in
                zip(body_parameters[i:i + incr_size], submodel_lrs)
            ])
        return param_lr_mappings, [plm['lr'] for plm in param_lr_mappings]

    def configure_optimizers(self):
        if not hasattr(self, 'train_data'):
            self.prepare_data()
        if type(self.lr) is float:
            optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.lr}])
        else:
            param_lr_mappings, self.lr = self.get_param_lr_maps(self.lr)
            optimizer = torch.optim.AdamW(param_lr_mappings)
        if self.use_one_cycle_lr_scheduler:
            return (
                [optimizer],
                [
                    torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, self.lr, epochs=self.max_epochs,
                        steps_per_epoch=int(np.ceil(len(self.trainer.datamodule.train_dataset) / self.batch_size)),
                        div_factor=1e2
                    )
                ]
            )
        elif self.lr_decay:
            return (
                [optimizer],
                [
                    torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=self.lr_decay_period, gamma=self.lr_decay_gamma
                    )
                ]
            )
        elif self.swa:
            if type(self.lr) is float:
                optimizer = torch.optim.SGD([{'params': self.model.parameters(), 'lr': self.lr}])
            else:
                param_lr_mappings, self.lr = self.get_param_lr_maps(self.lr)
                optimizer = torch.optim.SGD(param_lr_mappings)
            return [optimizer], [SWALR(optimizer, swa_lr=self.swa_lr)]
        else:
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='Simple1DCNN')
        parser.add_argument(
            '--logger_platform', type=str, choices=['tensorboard', 'wandb'], default='tensorboard'
        )
        parser.add_argument('--num_input_channels', type=int, default=12)
        parser.add_argument('--num_classes', type=int, default=5)
        parser.add_argument('--lr', nargs='+', type=float, default=[1e-3])
        parser.add_argument('--loss_function', type=str, default='bce_loss')
        parser.add_argument('--class_weighted_loss', type=bool, default=False)
        parser.add_argument('--focal_loss_alpha', type=float, default=1.0)
        parser.add_argument('--focal_loss_gamma', type=float, default=2.0)
        parser.add_argument('--use_one_cycle_lr_scheduler', type=bool, default=False)
        parser.add_argument('--lr_decay', type=bool, default=False)
        parser.add_argument('--lr_decay_period', type=int, default=5)
        parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
        parser.add_argument('--mixup', type=bool, default=False)
        parser.add_argument('--mixup_layer', type=int, default=0)
        parser.add_argument('--mixup_alpha', type=float, default=0.4)
        parser.add_argument('--swa', type=bool, default=False)
        parser.add_argument('--swa_lr', type=float, default=5e-2)
        parser.add_argument('--swa_epochs', type=int, default=10)
        parser.add_argument('--heatmap_layers', nargs='+', type=str, default=['pool3'])
        parser.add_argument('--show_heatmaps', type=bool, default=False)
        return parser
