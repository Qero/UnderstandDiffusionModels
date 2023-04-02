import torch
import numpy as np
import torchvision
from torch import nn
from torch import math
from matplotlib import pyplot as plt

from torchvision.transforms import transforms
from modules.lfw_dataset import LFWDataset

from modules.utils import tensor_to_pil_img


MEAN = np.array([0.66083133, 0.4900412, 0.40122637])
STD = np.array([0.18552901, 0.16933545, 0.1681688])


def show_once():
    dataset = LFWDataset(
        "dataset",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
            ]
        ),
    )

    img_idxes = [28, 56, 84, 112]
    fig, axes = plt.subplots(len(img_idxes), 2)
    for i, idx in enumerate(img_idxes):
        img, mixed, _, step = dataset[idx]
        show_img = tensor_to_pil_img(img, MEAN, STD)
        mixed = tensor_to_pil_img(mixed, MEAN, STD)

        axes[i][0].imshow(show_img)
        axes[i][0].set_title("Original")
        axes[i][1].imshow(mixed)
        axes[i][1].set_title(f"Mixed: {step}")
    plt.tight_layout()
    plt.savefig("outputs/once_forward_example.png")

def show_forward():
    dataset = LFWDataset(
        "dataset",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
            ]
        ),
    )

    img_idxes = [28, 56, 84, 112]
    steps = [250, 500, 750, 1000]
    fig, axes = plt.subplots(len(img_idxes), len(steps) + 1)
    for i, idx in enumerate(img_idxes):
        img = dataset.get_item_with_step(idx, 0)[0]
        show_img = tensor_to_pil_img(img, MEAN, STD)
        axes[i][0].imshow(show_img)
        axes[i][0].set_title("Step: 0")
        for j, step in enumerate(steps):
            mixed = dataset.get_item_with_step(idx, step)[1]
            mixed = tensor_to_pil_img(mixed, MEAN, STD)
            axes[i][j + 1].imshow(mixed)
            axes[i][j + 1].set_title(f"Step: {step}")
    plt.tight_layout()
    plt.savefig("outputs/forward_example.png")


if __name__ == """__main__""":
    show_once()
    show_forward()
