import argparse
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch

from core.analysis.visualization import compute_and_show_heatmap
from core.metrics import AUC, metric_summary
import core.models as ptbxl_models
from core.models import loss_functions
from core.models.wrappers import MetricModelWrapper, MixupWrapper, TTAWrapper


class PTBXLClassificationModel(LightningModule):
    """Lightning module wrapper for training a generic model for PTB-XL classification"""

    allowed_models = [
        'Simple1DCNN',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'wide_resnet50_2', 'wide_resnet101_2'
    ]
    allowed_loss_function_components = ['bce_loss', 'f1_loss', 'focal_loss']

    def __init__(self, model_name, *args, **kwargs):
        super(PTBXLClassificationModel, self).__init__()
        assert model_name in self.allowed_models, f"Please pick one of: {self.allowed_models}"
        self.model_name = model_name
        self.task_name = kwargs['task_name']
        self.logger_platform = kwargs['logger_platform']
        self.show_heatmaps = kwargs['show_heatmaps']
        self.data_dir = kwargs['data_dir']
        self.batch_size = kwargs['batch_size']
        self.num_input_channels = kwargs['num_input_channels']
        self.num_classes = kwargs['num_classes']
        self.num_workers = kwargs['num_workers']
        self.lr = sorted(kwargs['lr'])
        if len(self.lr) == 1:
            self.lr = self.lr[0]
        self.use_one_cycle_lr_scheduler = kwargs['use_one_cycle_lr_scheduler']
        self.max_epochs = kwargs['max_epochs']
        self.loss = self.get_loss_function(**kwargs)
        self.metric_functions = self.initialize_metric_functions()
        self.model = self.initialize_model(
            self.model_name, self.num_input_channels, self.num_classes,
            mixup=kwargs['mixup'], mixup_layer=kwargs['mixup_layer']
        )
        self.sampling_rate = kwargs['sampling_rate']
        self.save_hyperparameters(*kwargs.keys())

        self.train_step, self.val_step = 0, 0
        self.best_metrics = None

    def initialize_metric_functions(self):
        def acc(y_pred, y_true):
            return ((y_pred > 0.5) == y_true).sum().float() / (len(y_true) * len(y_true[0]))

        def auc(y_pred, y_true):
            return torch.Tensor([AUC(y_true.cpu().numpy(), y_pred.detach().cpu().numpy())])

        return {
            'loss': self.loss,
            'acc': acc,
            'auc': auc,
            'probs': lambda probs, targets: probs,
            'targets': lambda probs, targets: targets
        }

    def initialize_model(
        self,
        model_name,
        num_input_channels,
        num_classes,
        mixup=False,
        mixup_alpha=0.4,
        mixup_layer=0
    ):
        model = TTAWrapper(getattr(ptbxl_models, model_name)(
            num_input_channels=num_input_channels, num_classes=num_classes
        ))
        if mixup is False:
            return MetricModelWrapper(model, metric_functions=self.metric_functions)
        else:
            return MixupWrapper(
                model,
                metric_functions=self.metric_functions,
                mixup_layer=mixup_layer,
                alpha=mixup_alpha
            )

    def get_loss_function(self, loss_function='bce_loss', **kwargs):
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
                    assert factor in self.allowed_loss_function_components, \
                        f'Loss function term must be one of {self.allowed_loss_function_components}'
                    subloss_func = loss_functions.loss_function_factory(factor, **kwargs)

            def loss_component_function(y_pred, y_true):
                return float_factor * subloss_func(y_pred, y_true)

            loss_component_functions.append(loss_component_function)

        def combined_loss_function(y_pred, y_true):
            return sum([lcf(y_pred, y_true) for lcf in loss_component_functions])

        return combined_loss_function

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
                self.best_metrics['best_epoch'] = self.current_epoch
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output_metrics = self(batch)
        tensorboard_logs = {
            f'Train/train_step_{metric_name}': output_metrics[metric_name]
            for metric_name in output_metrics if metric_name not in ['probs', 'targets']
        }
        self.logger.log_metrics(tensorboard_logs, self.train_step)
        self.train_step += 1

        output_metrics['progress_bar'] = tensorboard_logs
        return output_metrics

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        acc_mean = torch.stack([x['acc'] for x in outputs]).mean()
        auc_mean = torch.stack([x['auc'] for x in outputs]).mean()
        probs = torch.cat([x['probs'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        (
            f_max,
            _,
            f_scores,
            average_precisions,
            average_recalls,
            thresholds
        ) = metric_summary(targets.numpy(), probs.numpy())

        self.logger.log_metrics(
            {
                "Train/train_loss": loss_mean,
                "Train/train_acc": acc_mean,
                "Train/train_auc": auc_mean
            },
            self.current_epoch + 1
        )
        try:
            self.logger.add_scalars(
                "Losses", {"train_loss": loss_mean}, self.current_epoch + 1
            )
            self.logger.add_scalars(
                "Accuracies", {"train_acc": acc_mean}, self.current_epoch + 1
            )
            self.logger.add_scalars(
                "AUCs", {"train_auc": auc_mean}, self.current_epoch + 1
            )
            self.logger.add_scalars(
                "F1 Max Scores", {"train_f1-max": f_max}, self.current_epoch + 1
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
        self.logger.log_metrics(metric_appendix, self.current_epoch + 1)
        return {
            'train_epoch_loss': loss_mean,
            'train_epoch_acc': acc_mean,
            'train_epoch_auc': auc_mean,
            'train_f_max': f_max,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output_metrics = self(batch)
        tensorboard_logs = {
            f'Validation/val_step_{metric_name}': output_metrics[metric_name]
            for metric_name in output_metrics if metric_name not in ['probs', 'targets']
        }
        self.logger.log_metrics(tensorboard_logs, self.val_step)
        self.val_step += 1

        output_metrics['progress_bar'] = tensorboard_logs
        return output_metrics

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['acc'] for x in outputs]).mean()
        val_auc_mean = torch.stack([x['auc'] for x in outputs]).mean()
        probs = torch.cat([x['probs'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        (
            f_max,
            _,
            f_scores,
            average_precisions,
            average_recalls,
            thresholds
        ) = metric_summary(targets.numpy(), probs.numpy())

        self.logger.log_metrics(
            {
                "Validation/val_loss": val_loss_mean,
                "Validation/val_acc": val_acc_mean,
                "Validation/val_auc": val_auc_mean,
            },
            self.current_epoch + 1
        )
        try:
            self.logger.add_scalars(
                "Losses", {"val_loss": val_loss_mean}, self.current_epoch + 1
            )
            self.logger.add_scalars(
                "Accuracies", {"val_acc": val_acc_mean}, self.current_epoch + 1
            )
            self.logger.add_scalars(
                "AUCs", {"val_auc": val_auc_mean}, self.current_epoch + 1
            )
            self.logger.add_scalars(
                "F1 Max Scores", {"val_f1-max": f_max}, self.current_epoch + 1
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
        self.logger.log_metrics(metric_appendix, self.current_epoch + 1)
        self.update_hyperparams_and_metrics(
            {
                'val_epoch_loss': val_loss_mean,
                'val_epoch_acc': val_acc_mean,
                'val_epoch_auc': val_auc_mean,
                'val_epoch_f_max': f_max
            }
        )
        return {
            'val_epoch_loss': val_loss_mean,
            'val_epoch_acc': val_acc_mean,
            'val_epoch_auc': val_auc_mean,
            'val_epoch_f_max': f_max
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        output_metrics = self(batch)
        tensorboard_logs = {
            f'Test/test_step_{metric_name}': output_metrics[metric_name]
            for metric_name in output_metrics if metric_name not in ['probs', 'targets']
        }
        output_metrics['progress_bar'] = tensorboard_logs
        return output_metrics

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['acc'] for x in outputs]).mean()
        test_auc_mean = torch.stack([x['auc'] for x in outputs]).mean()
        probs = torch.cat([x['probs'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        if self.show_heatmaps:
            print(f'Logging heatmaps - this might take some time...')
            with torch.set_grad_enabled(True):
                self.visualize_best_and_worst_heatmaps(probs.numpy(), targets.numpy())
            print(f'...done logging heatmaps.')
        f_max = metric_summary(targets.numpy(), probs.numpy())[0]
        self.logger.plot_confusion_matrix(targets, (probs > 0.5).cpu().numpy(), self.labels)
        self.logger.plot_roc(targets.long().cpu().numpy(), probs.cpu().numpy(), self.labels)
        return {
            'log': {
                'test_epoch_loss': test_loss_mean,
                'test_epoch_acc': test_acc_mean,
                'test_epoch_auc': test_auc_mean,
                'test_epoch_f_max': f_max
            }
        }

    def visualize_best_and_worst_heatmaps(self, test_probs, test_targets):
        label_mapping = {k: i for i, k in enumerate(self.labels)}
        test_diffs = np.abs(test_probs - test_targets)
        for label_name, label_index in label_mapping.items():
            test_label_targets = test_targets[:, label_index]

            test_label_diffs = test_diffs[:, label_index].copy()
            # Best positives, worst positives
            positive_mask = test_label_targets == 1
            test_label_diffs[~positive_mask] = 1.1
            best_indices = test_label_diffs.argsort()[:3]
            test_label_diffs[~positive_mask] = -0.1
            test_label_diffs *= -1
            worst_indices = test_label_diffs.argsort()[:3]
            best_test_positive_datapoints = [self.val_dataset[i] for i in best_indices]
            worst_test_positive_datapoints = [self.val_dataset[i] for i in worst_indices]

            test_label_diffs = test_diffs[:, label_index].copy()
            # Best negatives, worst negatives
            negative_mask = test_label_targets == 0
            test_label_diffs[~negative_mask] = 1.1
            best_indices = test_label_diffs.argsort()[:3]
            test_label_diffs[~negative_mask] = -0.1
            test_label_diffs *= -1
            worst_indices = test_label_diffs.argsort()[:3]
            best_test_negative_datapoints = [self.val_dataset[i] for i in best_indices]
            worst_test_negative_datapoints = [self.val_dataset[i] for i in worst_indices]

            best_test_positive_figs = [
                compute_and_show_heatmap(
                    self.model.model, 'pool3',
                    datapoint, label_mapping, class_of_concern=label_name,
                    show_fig=False, figsize=(10, 40)
                ) for datapoint in best_test_positive_datapoints
            ]
            self.logger.plot_figures({
                f'{label_name} Present/Best (Rank {i})': fig
                for i, fig in enumerate(best_test_positive_figs)
            })
            worst_test_positive_figs = [
                compute_and_show_heatmap(
                    self.model.model, 'pool3',
                    datapoint, label_mapping, class_of_concern=label_name,
                    show_fig=False, figsize=(10, 40)
                ) for datapoint in worst_test_positive_datapoints
            ]
            self.logger.plot_figures({
                f'{label_name} Present/Worst (Rank {i})': fig
                for i, fig in enumerate(worst_test_positive_figs)
            })
            best_test_negative_figs = [
                compute_and_show_heatmap(
                    self.model.model, 'pool3',
                    datapoint, label_mapping, class_of_concern=label_name,
                    show_fig=False, figsize=(10, 40)
                ) for datapoint in best_test_negative_datapoints
            ]
            self.logger.plot_figures({
                f'{label_name} Absent/Best (Rank {i})': fig
                for i, fig in enumerate(best_test_negative_figs)
            })
            worst_test_negative_figs = [
                compute_and_show_heatmap(
                    self.model.model, 'pool3',
                    datapoint, label_mapping, class_of_concern=label_name,
                    show_fig=False, figsize=(10, 40)
                ) for datapoint in worst_test_negative_datapoints
            ]
            self.logger.plot_figures({
                f'{label_name} Absent/Worst (Rank {i})': fig
                for i, fig in enumerate(worst_test_negative_figs)
            })

    def get_param_lr_maps(self, lrs):
        """ Output parameter LR mappings for setting up an optimizer for `model`."""
        body_parameters = [param for (_, param) in self.model.model.named_parameters()]
        param_lr_mappings = []
        incr_size = len(body_parameters) // (len(lrs) - 1)
        for i in range(0, len(body_parameters), incr_size):
            submodel_lrs = np.geomspace(
                lrs[i // incr_size], lrs[(i // incr_size) + 1],
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
                        steps_per_epoch=int(np.ceil(len(self.train_dataset) / self.batch_size)),
                        div_factor=1e2
                    )
                ]
            )
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
        parser.add_argument('--focal_loss_alpha', type=float, default=1.0)
        parser.add_argument('--focal_loss_gamma', type=float, default=2.0)
        parser.add_argument('--use_one_cycle_lr_scheduler', type=bool, default=False)
        parser.add_argument('--mixup', type=bool, default=False)
        parser.add_argument('--mixup_layer', type=int, default=0)
        parser.add_argument('--mixup_alpha', type=float, default=0.4)
        return parser
