import torch
from random import randint

import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from modules.linear_diffuser import LinearDifuser


class LFWDataset(Dataset):
    def __init__(self, path, transform=None):
        self._path = path
        self._transform = transform
        if self._transform is None:
            self._transform = transforms.Compose([transforms.ToTensor()])

        self._data = torchvision.datasets.LFWPeople(root="dataset", download=True)
        self._diffuser = LinearDifuser()

    def __len__(self):
        return len(self._data)

    def get_item_with_step(self, idx, step):
        ori, _ = self._data[idx]
        ori = self._transform(ori)
        mixed, noise = self._diffuser.diffuse(step, ori)
        return ori, mixed, noise

    def __getitem__(self, idx):
        step = randint(1, self._diffuser.max_step)
        ori, mixed, noise = self.get_item_with_step(idx, step)
        return ori, mixed, noise, step
