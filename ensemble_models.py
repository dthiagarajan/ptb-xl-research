""" Gets ensembled predictions for classification models trained on the PTB-XL dataset.

Example usage:
python ensemble_models.py --versions 0 1 2
"""
import argparse
from pathlib import Path
import pickle
from pytorch_lightning import seed_everything, Trainer
import shutil
import torch

from core.data import PTBXLDataModule
from core.loggers import TensorBoardLogger, WandbLogger
from core.metrics import metric_summary
from core.models import PTBXLClassificationModel


def get_args():
    """Argument parser for ensembling trained PTB-XL classification models.

    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Ensemble trained classification models on PTB-XL diagnostic superclass'
    )
    parser.add_argument('--log_dir', type=str, default='./completed_runs')
    parser.add_argument(
        '--individual_model_log_dir', type=str, default='./ensemble_logs/individual_logs'
    )
    parser.add_argument('--use_cached_outputs', action='store_true')
    parser.add_argument('--model_outputs_fp', type=str, default='test_outputs.pkl')
    parser.add_argument('--versions', nargs='+', type=int, default=[0])
    parser = PTBXLClassificationModel.add_model_specific_args(parser)
    parser = PTBXLDataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


def version_directory(ensemble_log_dir):
    version_cnt = 0
    filedir = Path(ensemble_log_dir, f'version_{version_cnt}')
    while Path(ensemble_log_dir, f'version_{version_cnt}').is_dir():
        filedir = Path(ensemble_log_dir, f'version_{version_cnt}')
        version_cnt += 1
    return filedir


def ensemble_outputs(all_test_outputs, model):
    ensemble_acc = torch.stack([x['acc'] for outputs in all_test_outputs for x in outputs]).mean()
    ensemble_auc = torch.stack([x['auc'] for outputs in all_test_outputs for x in outputs]).mean()
    ensemble_probs = torch.stack(
        [torch.cat([x['probs'] for x in outputs]).cpu() for outputs in all_test_outputs],
        dim=-1
    ).mean(-1)
    targets = torch.cat([x['targets'] for x in all_test_outputs[0]]).cpu()
    ensemble_f_max, ensemble_auc, _, _, _, _ = metric_summary(
        targets.cpu().numpy(), ensemble_probs.cpu().numpy()
    )
    model.logger.plot_confusion_matrix(targets, (ensemble_probs > 0.5).cpu().numpy(), model.labels)
    model.logger.plot_roc(targets.long().cpu().numpy(), ensemble_probs.cpu().numpy(), model.labels)
    metrics = {
        'acc': ensemble_acc,
        'auc': ensemble_auc,
        'f_max': ensemble_f_max
    }
    model.logger.log_hyperparams(model.hparams, metrics)
    return metrics


if __name__ == '__main__':
    args = get_args()
    seed_everything(31)

    ensemble_version_directory = version_directory('./ensemble_logs')
    Path(args.individual_model_log_dir).mkdir(exist_ok=True)

    data_module = PTBXLDataModule(
        args.data_dir, args.sampling_rate, args.task_name,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    data_module.setup()

    test_outputs = []
    for version in args.versions:
        model_checkpoint = sorted(list(Path(args.log_dir, f'version_{version}').glob('*.ckpt')))[-1]
        model_outputs_fp = Path(args.log_dir, f'version_{version}', args.model_outputs_fp)
        if args.use_cached_outputs is True and model_outputs_fp.exists():
            print(f'Loading previously computed test outputs from {model_outputs_fp}...')
            with open(model_outputs_fp, 'rb') as f:
                next_test_outputs = pickle.load(f)
        else:
            print(f'Loading model from {model_checkpoint}...')
            model = PTBXLClassificationModel.load_from_checkpoint(str(model_checkpoint))
            print(f'...done. Loaded {model.model_name} from {model_checkpoint}.')

            if args.logger_platform == 'wandb':
                logger = WandbLogger(project="ptb-xl")
            elif args.logger_platform == 'tensorboard':
                logger = TensorBoardLogger(args.individual_model_log_dir, name='')
                model.log_dir = args.individual_model_log_dir

            trainer = Trainer.from_argparse_args(args, deterministic=True, logger=logger)

            model.labels = data_module.labels  # Needed for confusion matrix labels
            print(f'Running inference for model {model.model_name}, version {version}...')
            trainer.test(model=model, datamodule=data_module)
            print(f'...done. Saving test outputs to {model_outputs_fp}...')
            next_test_outputs = model.test_outputs
            with open(model_outputs_fp, 'wb') as f:
                pickle.dump(next_test_outputs, f)
        test_outputs.append(next_test_outputs)
        print(f'...done.')

    shutil.rmtree(Path(args.individual_model_log_dir))
    output_fp = Path(
        './ensemble_logs', f"ensembled_{'-'.join([str(v) for v in args.versions])}.pkl"
    )
    print(f'Dumping all outputs in a list to {output_fp}...')
    with open(output_fp, 'wb') as f:
        pickle.dump(test_outputs, f)
    print(f'...done.')

    model = PTBXLClassificationModel(**vars(args))
    model.labels = data_module.labels  # Needed for confusion matrix labels
    if args.logger_platform == 'wandb':
        model.logger = WandbLogger(project="ptb-xl")
    elif args.logger_platform == 'tensorboard':
        model.logger = TensorBoardLogger('./ensemble_logs', name='')
        model.log_dir = './ensemble_logs'
    ensemble_outputs(test_outputs, model)
