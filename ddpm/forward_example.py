import numpy as np
from matplotlib import pyplot as plt

from torchvision.transforms import transforms
from modules.flowers102_dataset import Flowers102Dataset

from modules.utils import tensor_to_pil_img


# for flowers102
MEAN = np.array([0.4906, 0.4362, 0.4803])
STD = np.array([0.2542, 0.2241, 0.2903])


def show_forward():
    dataset = Flowers102Dataset(
        "dataset",
        max_Step=1000,
        transform=transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
            ]
        ),
    )

    img_idxes = [28, 56, 84, 112]
    steps = [0, 250, 500, 750, 1000]
    fig, axes = plt.subplots(len(img_idxes), len(steps))
    for i, idx in enumerate(img_idxes):
        for j, step in enumerate(steps):
            mixed = dataset[idx*(dataset.max_step+1) + step][1]
            mixed = tensor_to_pil_img(mixed, MEAN, STD)
            axes[i][j].imshow(mixed)
            axes[i][j].set_title(f"Step: {step}")
    plt.tight_layout()
    plt.savefig("outputs/forward_example.png")


if __name__ == """__main__""":
    show_forward()
