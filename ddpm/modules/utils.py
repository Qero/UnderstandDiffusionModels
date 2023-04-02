import torch
from torchvision import transforms


def tensor_to_pil_img(img, mean, std):
    img = torch.permute(img, (1, 2, 0)) * std + mean
    img = torch.permute(img, (2, 0, 1))
    img = transforms.ToPILImage()(img)
    return img