import pytest
import torch

from core.callbacks import MultiMetricModelCheckpoint


class MockTrainer:
    def __init__(self):
        self.global_rank = 0
        self.callback_metrics = {'val_loss': 1.5, 'val_acc': 0.5}
        self.current_epoch = 1


@pytest.fixture()
def callback():
    return MultiMetricModelCheckpoint(
        verbose=True, monitors=['val_loss', 'val_acc'], modes=['min', 'max']
    )


class TestMultiMetricModelCheckpoint:
    def test_check_monitor_top_k(self, callback):
        assert callback.check_monitor_top_k({'val_loss': 1.5, 'val_acc': 0.5})
        callback.best_k_models = {
            'dummy_path': {'val_loss': torch.tensor(1.5), 'val_acc': torch.tensor(0.5)}
        }
        callback.kth_best_model_path = 'dummy_path'
        assert callback.check_monitor_top_k({'val_loss': 1.0, 'val_acc': 1.0}) == torch.tensor(1)
        assert callback.check_monitor_top_k({'val_loss': 2.0, 'val_acc': 0.1}) == torch.tensor(0)

    def test_on_validation_end(self, monkeypatch, callback):
        callback.filename = '{epoch}'
        callback.dirpath = './'

        def mock_save_model(fp):
            self.output_fp = fp

        monkeypatch.setattr(callback, "_save_model", mock_save_model)
        trainer = MockTrainer()
        callback.on_validation_end(trainer, None)
        assert self.output_fp == './epoch=1.ckpt'

    def test_do_check_save(self, monkeypatch, callback):
        def mock_save_model(fp):
            self.output_fp = fp

        monkeypatch.setattr(callback, "_save_model", mock_save_model)

        callback._do_check_save('./dummy.pth', {'val_loss': 1.5, 'val_acc': 0.5}, 0)
        assert self.output_fp == './dummy.pth'
        callback._do_check_save('./overwritten.pth', {'val_loss': 2.0, 'val_acc': 0.0}, 0)
        assert self.output_fp == './overwritten.pth'
