import math

import torch
from torch import nn
from modules.residual_unet import ResUNet


class DiffusionModel(nn.Module):
    def __init__(self, max_step, step_embedding_dims, img_size, blocks_channels, block_depth):
        super().__init__()
        self._max_step = max_step
        self._step_embedding_dims = step_embedding_dims
        self._img_size = img_size
        self._blocks_channels = blocks_channels
        self._block_depth = block_depth

        # define step embedding    
        frequencies = torch.exp(
            torch.linspace(
                math.log(1),
                math.log(self._max_step),
                self._step_embedding_dims // 2,
            )
        )
        self._step_angular_speeds = 2.0 * torch.pi * frequencies
        self._step_angular_speeds = self._step_angular_speeds.unsqueeze(0)
        self._emb_upsample = torch.nn.UpsamplingNearest2d(size=self._img_size)

        # define inner network
        self._net = ResUNet(
            in_channels=3 + self._step_embedding_dims,
            out_channels=3,
            blocks_channels=self._blocks_channels,
            block_depth=self._block_depth
        )

    def _step_embedding(self, step):
        self._step_angular_speeds = self._step_angular_speeds.to(step.device)
        emb = torch.concat([
            torch.sin(self._step_angular_speeds * step), 
            torch.cos(self._step_angular_speeds * step)], axis=-1
        )
        return emb

    def forward(self, x, step):
        step = step.unsqueeze(-1)
        step_emb = self._step_embedding(step).unsqueeze(-1).unsqueeze(-1)
        step_emb = self._emb_upsample(step_emb)
        x = torch.cat([x, step_emb], dim=1)
        y = self._net(x)
        return y


if __name__ == '''__main__''':
    # test diffusion model
    model = DiffusionModel(max_step=1000,
                           step_embedding_dims=16,
                           img_size=32,
                           blocks_channels=[32, 64, 128, 256],
                           block_depth=2)
    model.cuda()
    x = torch.randn(2, 3, 32, 32).cuda()
    step = torch.randint(1, 1000, (2, 1))
    y = model(x, step)
    print(y.shape)
    print(y)
