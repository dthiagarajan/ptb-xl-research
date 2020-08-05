import torch.nn as nn


class Simple1DCNN(nn.Module):
    """Simple 1D CNN for processing signals. Organized as

    Simple1DCNN(
        (layer1): Conv1d(12, 32, kernel_size=(3,), stride=(1,))
        (pool1): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
        (bn1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (layer2): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
        (pool2): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
        (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (layer3): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
        (pool3): AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,))
        (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (final_pool): AdaptiveAvgPool1d(output_size=1)
    )

    """
    def __init__(self, num_input_channels=12, num_classes=5):
        super(Simple1DCNN, self).__init__()
        self.layer1 = nn.Conv1d(num_input_channels, 32, kernel_size=3)
        self.pool1 = nn.AvgPool1d(3)
        self.bn1 = nn.BatchNorm1d(32)
        self.layer2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.AvgPool1d(3)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool3 = nn.AvgPool1d(3)
        self.bn3 = nn.BatchNorm1d(128)

        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.final_layer = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = nn.functional.relu(out)
        out = self.bn1(out)

        out = self.layer2(out)
        out = self.pool2(out)
        out = nn.functional.relu(out)
        out = self.bn2(out)

        out = self.layer3(out)
        out = self.pool3(out)
        out = nn.functional.relu(out)
        out = self.bn3(out)

        out = self.final_pool(out)
        out = self.final_layer(out)
        return out.view(x.size(0), -1)
