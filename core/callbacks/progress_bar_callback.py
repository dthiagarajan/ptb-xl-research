
from pytorch_lightning.callbacks import ProgressBar


def convert_inf(x):
    """ The tqdm doesn't support inf values. We have to convert it to None. """
    if x == float('inf'):
        return None
    return x


class ProgressBar(ProgressBar):
    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float('inf') and not trainer.fast_dev_run:
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(convert_inf(total_batches))
        self.main_progress_bar.set_description(f'Epoch {pl_module.training_log_epoch}')
