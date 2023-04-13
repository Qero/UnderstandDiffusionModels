import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self._residual_conv = nn.Identity()
        if in_channels != out_channels:
            self._residual_conv = nn.Conv2d(in_channels, out_channels, 1)

        self._bn1 = nn.BatchNorm2d(in_channels)
        self._conv1 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self._act1 = nn.SiLU()
        self._bn2 = nn.BatchNorm2d(out_channels)
        self._conv2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self._act2 = nn.SiLU()
        self._conv_block = nn.Sequential(
            self._bn1,
            self._conv1,
            self._act1,
            self._bn2,
            self._conv2,
            self._act2,
        )

    def forward(self, x):
        r = self._residual_conv(x)
        x = self._conv_block(x)
        x = x + r
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth) -> None:
        super().__init__()

        self._blocks = nn.ModuleList()
        for _ in range(depth):
            self._blocks.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels

        self._down_sample = nn.AvgPool2d(2)

    def forward(self, x, skips):
        for block in self._blocks:
            x = block(x)
            skips.append(x)
        x = self._down_sample(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth) -> None:
        super().__init__()

        self._up_sample = nn.Upsample(scale_factor=2, mode="bilinear")

        self._blocks = nn.ModuleList()
        for _ in range(depth):
            self._blocks.append(ResidualBlock(in_channels + out_channels, out_channels))
            in_channels = out_channels

    def forward(self, x, skips):
        x = self._up_sample(x)
        for block in self._blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks_channels, block_depth) -> None:
        super().__init__()

        self._down_blocks = nn.ModuleList()
        self._bridge = nn.Sequential()
        self._up_blocks = nn.ModuleList()

        last_channels = in_channels
        for block_channels in blocks_channels[:-1]:
            down_block = DownBlock(last_channels, block_channels, block_depth)
            self._down_blocks.append(down_block)
            last_channels = block_channels

        block_channels = blocks_channels[-1]
        for _ in range(block_depth):
            self._bridge.append(ResidualBlock(last_channels, block_channels))
            last_channels = block_channels

        for block_channels in reversed(blocks_channels[:-1]):
            up_block = UpBlock(last_channels, block_channels, block_depth)
            self._up_blocks.append(up_block)
            last_channels = block_channels

        self._final_conv = nn.Conv2d(last_channels, out_channels, 1, padding="same")
        self._final_act = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for down_block in self._down_blocks:
            x = down_block(x, skips)
        x = self._bridge(x)
        for up_block in self._up_blocks:
            x = up_block(x, skips)
        x = self._final_conv(x)
        # y = self._final_act(x)
        return x


# Test ResUNet
if __name__ == "__main__":
    model = ResUNet(3, 3, [32, 64, 128, 256], 2).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    y = model(x)
    print(y.device)
    print(y.shape)
    assert y.shape == torch.Size([1, 3, 256, 256])
