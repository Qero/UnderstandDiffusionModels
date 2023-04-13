

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.animation import FuncAnimation

from diffusion_model import DiffusionModel
from modules.utils import tensor_to_pil_img


MEAN = np.array([0.4906, 0.4362, 0.4803])
STD = np.array([0.2542, 0.2241, 0.2903])


def linear_schedule(step, min_variance=10**-4, max_variance=0.02, max_step=1000):
    noise_variance = torch.linspace(min_variance, max_variance, max_step)
    tilde_alpha = torch.cumprod(1 - noise_variance, dim=0)
    signal_rate = torch.sqrt(tilde_alpha[step])
    noise_rate = torch.sqrt(1 - tilde_alpha[step])
    return signal_rate, noise_rate


def show_once():
    states = torch.load("./checkpoints/diffusion_model45.pt", map_location="cpu")
    model_state_dict = states["model_state_dict"]

    model = DiffusionModel(
        max_step=1000,
        step_embedding_dims=32,
        img_size=128,
        blocks_channels=[32, 64, 96, 128],
        block_depth=2
    )
    model.load_state_dict(model_state_dict)
    model.cpu()
    model.eval()

    fig, axes = plt.subplots(1, 5)
    mixed = torch.normal(0, 1, (1, 3, 128, 128)).cpu()
    pred_img = torch.permute(mixed.cpu()[0], (1, 2, 0)).detach().numpy()
    pred_img = (pred_img * STD + MEAN) * 255
    pred_img = pred_img.clip(min=0, max=255).astype(np.uint8)
    axes_idx = 0
    axes[axes_idx].imshow(pred_img)
    axes[axes_idx].set_title("Start")
    axes_idx += 1
    with torch.no_grad(): 
        for step in range(1000, 0, -1):
            pred_noise = model(mixed, torch.tensor([step]))

            signal_rate, noise_rate = linear_schedule(step-1)
            pred_img = (mixed - noise_rate * pred_noise) / signal_rate

            next_signal_rate, next_noise_rate = linear_schedule(step-2)
            mixed = next_signal_rate * pred_img + next_noise_rate * pred_noise

            if step in (751, 501, 251, 1):
                pred_img = torch.permute(pred_img.cpu()[0], (1, 2, 0)).detach().numpy()
                pred_img = (pred_img * STD + MEAN) * 255
                pred_img = pred_img.clip(min=0, max=255).astype(np.uint8)
                axes[axes_idx].imshow(pred_img)
                axes[axes_idx].set_title(f"Step: {step-1}")
                axes_idx += 1

    plt.tight_layout()
    plt.savefig("outputs/show_results.png")


def show_multi():
    states = torch.load("./checkpoints/diffusion_model45.pt", map_location="cpu")
    model_state_dict = states["model_state_dict"]

    model = DiffusionModel(
        max_step=1000,
        step_embedding_dims=32,
        img_size=128,
        blocks_channels=[32, 64, 96, 128],
        block_depth=2
    )
    model.load_state_dict(model_state_dict)
    model.cpu()
    model.eval()

    def _3x3_grid(img_tensors):
        # NHWC to List of CHW
        img_rows = torch.split(img_tensors, 3, dim=0)
        # list of HWC to list of 3x3 grid
        img_rows = [torch.cat([img for img in img_row], dim=2) for img_row in img_rows]
        img = torch.cat(img_rows, dim=1)
        return img
    
    imgs = []
    mixed = torch.normal(0, 1, (9, 3, 128, 128)).cpu()
    imgs.append(tensor_to_pil_img(_3x3_grid(mixed), MEAN, STD))
    with torch.no_grad():
        for step in range(1000, 0, -1):
            pred_noise = model(mixed, torch.ones(9) * step)

            signal_rate, noise_rate = linear_schedule(step-1)
            pred_img = (mixed - noise_rate * pred_noise) / signal_rate

            next_signal_rate, next_noise_rate = linear_schedule(step-2)
            mixed = next_signal_rate * pred_img + next_noise_rate * pred_noise
            print(step)
            if step % 24 == 0 or step in (1000, 1):
                imgs.append(tensor_to_pil_img(_3x3_grid(pred_img), MEAN, STD))

    # show imgs in an animation
    fig = plt.figure()
    im = plt.imshow(imgs[0])
    plt.show()

    def _updatefig(i):
        im.set_array(imgs[i])
        return im,

    ani = FuncAnimation(fig, _updatefig, frames=len(imgs), interval=60, blit=True)
    ani.save("outputs/show_results.gif", writer="imagemagick")


if __name__ == '''__main__''':
    show_once()
    # show_multi()
