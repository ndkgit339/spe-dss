# Reference:
#   - github link: https://github.com/tarepan/Scyclone-PyTorch


import torch
from torch import nn
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    def __init__(self, channel, kernel_size, negative_slope):
        super().__init__()

        # blocks
        self.conv_block = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope),
            nn.Conv1d(channel, channel, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Scyclone_G(nn.Module):
    """
    Scyclone Generator
    """

    def __init__(
        self, 
        in_channel=128,
        mid_channel=256,
        out_channel=128,
        n_resblock=7,
        resblock_kernel_size=5,
        negative_slope=0.01,
        optuna_trial=None
    ):
        super().__init__()

        if optuna_trial is not None:
            mid_channel = optuna_trial.suggest_int("mid_channel", in_channel, in_channel*3)
            n_resblock = optuna_trial.suggest_int("n_resblock", 2, 7)

        blocks = [
            nn.Conv1d(in_channel, mid_channel, 1),
            nn.LeakyReLU(negative_slope),
        ]

        blocks += [
            ResidualBlock(mid_channel, resblock_kernel_size, negative_slope) 
            for _ in range(n_resblock)
        ]

        blocks += [
            nn.Conv1d(mid_channel, out_channel, 1),
            nn.LeakyReLU(negative_slope),
        ]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Scyclone_D_Block(nn.Module):
    def __init__(self, hidden_channels=1026):
        super().__init__()

        layers = [
            spectral_norm(
                nn.Conv1d(
                    in_channels=hidden_channels, out_channels=hidden_channels,
                    kernel_size=5, stride=1, padding=2)
                ),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(
                nn.Conv1d(
                    in_channels=hidden_channels, out_channels=hidden_channels,
                    kernel_size=5, stride=1, padding=2)
                ),            
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.model(x)


class Scyclone_D(nn.Module):
    def __init__(self, in_channels=513, hidden_channels=1026, n_blocks=6):
        super().__init__()

        layers = [
            spectral_norm(
                nn.Conv1d(
                    in_channels=in_channels, out_channels=hidden_channels,
                    kernel_size=1, stride=1)
                ),
            nn.LeakyReLU(negative_slope=0.2),
            ]

        layers += [Scyclone_D_Block(hidden_channels=hidden_channels)     
                   for _ in range(n_blocks)]

        layers += [
            spectral_norm(
                nn.Conv1d(
                    in_channels=hidden_channels, out_channels=1,
                    kernel_size=1, stride=1)
                ),
            nn.LeakyReLU(negative_slope=0.2),
            ]

        layers += [
            nn.AvgPool1d(1000)
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__=='__main__':

    model = Scyclone_G(
        in_channel=80, mid_channel=40, out_channel=1, n_resblock=4,
        resblock_kernel_size=3, negative_slope=0.01)
    x = torch.randn(32,80,1000)
    y = model(x)
    print(x.shape)
    print(y.shape)