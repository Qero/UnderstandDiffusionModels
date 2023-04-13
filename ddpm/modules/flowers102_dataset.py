import torch
from random import randint

import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from modules.linear_diffuser import LinearDifuser


class Flowers102Dataset(Dataset):
    def __init__(self, path, max_Step, transform=None):
        self._path = path
        self._max_step = max_Step
        self._transform = transform
        if self._transform is None:
            self._transform = transforms.Compose([transforms.ToTensor()])

        self._data = torchvision.datasets.Flowers102(root="dataset", download=True)
        self._ori_size = len(self._data)
        self._diffuser = LinearDifuser(max_step=self._max_step)

    @property
    def max_step(self):
        return self._max_step

    @property
    def ori_size(self):
        return self._ori_size

    def __len__(self):
        return self._ori_size * (self._max_step+1)

    def __getitem__(self, i):
        idx = i // (self._max_step + 1)
        step = i % (self._max_step + 1)
        ori_img, _ = self._data[idx]
        ori_img = self._transform(ori_img)
        mixed_img, noise = self._diffuser.diffuse(step, ori_img)
        return ori_img, mixed_img, noise, step
