import torch
from torch import nn
from torch import math


class LinearDifuser(nn.Module):
    def __init__(
        self, max_step=1000, min_variance=10**-4, max_variance=0.02
    ) -> None:
        self._max_step = max_step

        # beta[1...T] in paper
        self._noise_variance = torch.linspace(min_variance, max_variance, max_step)

        # alpha[t] = 1 - beta[t] in paper
        # \tilde{alpha}[t] = alpha[0]*alpha[1]*...*alpha[t] in paper
        self._tilde_alpha = torch.cumprod(1 - self._noise_variance, dim=0)

    def _schedule(self, step):
        assert step > 0
        assert step < self._max_step
        signal_rate = math.sqrt(self._tilde_alpha[step])
        noise_rate = math.sqrt(1 - self._tilde_alpha[step])
        return signal_rate, noise_rate

    @property
    def max_step(self):
        return self._max_step

    def diffuse(self, step, img):
        if step == 0:
            signal_rate = 1
            noise_rate = 0
        else:
            signal_rate, noise_rate = self._schedule(step - 1)
        noise = torch.normal(0, 1, img.shape)
        mixed = signal_rate * img + noise_rate * noise
        return mixed, noise