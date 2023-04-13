
import time
import argparse
import torch.multiprocessing as mp

import torch
import numpy as np
from torchvision import transforms
import tensorflow_io as tfio
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.multi_gpu_traning_state import MultiGPUTraningState
from modules.flowers102_dataset import Flowers102Dataset
from diffusion_model import DiffusionModel


MEAN = np.array([0.4906, 0.4362, 0.4803])
STD = np.array([0.2542, 0.2241, 0.2903])

def init_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_sample_step', type=int, default=1000)

    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument("--data_dir", type=str, help="the dir to data files.")
    parser.add_argument("--log_dir", type=str, help="the dir to training log.")
    parser.add_argument("--model_dir", type=str, help="the dir to save model checkpoints.")

    parser.add_argument("--num_loaders", type=int, help="the number of data loading processions.", default=4)
    parser.add_argument("--num_gpu", type=int, help="the number of training GPUs.", default=1)
    parser.add_argument("--num_epoch", type=int, help="the training epoch.", default=100)
    args = parser.parse_args()
    return args


def data_ready(args):
    dataset = Flowers102Dataset(
        args.data_dir,
        max_Step=args.max_sample_step,
        transform=transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MEAN,
                    std=STD,
                ),
            ]
        ),
    )
    return dataset


def linear_schedule(step, min_variance=10**-4, max_variance=0.02, max_step=1000):
    noise_variance = torch.linspace(min_variance, max_variance, max_step)
    tilde_alpha = torch.cumprod(1 - noise_variance, dim=0)
    signal_rate = torch.sqrt(tilde_alpha[step])
    noise_rate = torch.sqrt(1 - tilde_alpha[step])
    return signal_rate, noise_rate


def train(rank, args):
    dataset = data_ready(args)

    with MultiGPUTraningState(rank, args.num_gpu, args.model_dir, "diffusion_model", args.log_dir, max_saving_num=4) as ts:
        device_ids = [rank]

        # 1. 定义数据读取
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_loaders)
        
        # 2. 定义模型
        model = DiffusionModel(max_step=args.max_sample_step,
                                step_embedding_dims=32,
                                img_size=args.img_size,
                                blocks_channels=[32, 64, 96, 128],
                                block_depth=2).cuda()
        model = DDP(
            model,
            device_ids=device_ids,
            output_device=rank,
            find_unused_parameters=True,
        )

        # 3. 定义Loss函数
        criterion = torch.nn.L1Loss()

        # 4. 定义优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if ts.optim_state_dict is not None:
            optimizer.load_state_dict(ts.optim_state_dict)
        
        # 5. 定义rainin_states(全局步、历史损失)
        training_states = {}
        training_states["global_step"] = 0
        training_states["losses"] = []
        if ts.training_states is not None:
            # 5.1 断点状态下，恢复训练状态，并重画看板
            training_states = ts.training_states
            for step in range(0, training_states["global_step"]):
                ts.write(
                    "add_scalar",
                    tag="Loss/training_loss",
                    scalar_value=training_states["losses"][step],
                    global_step=step,
                )

        # 6. 训练
        init_epoch = ts.init_epoch
        for epoch in range(init_epoch + 1, init_epoch + args.num_epoch):
            model.train()
            for i, data in enumerate(dataloader):
                _, mixeds, noises, step = data
                mixeds = mixeds.cuda()
                noises = noises.cuda()
                step = step.cuda()

                predictions = model(mixeds, step)
                loss = criterion(predictions, noises)
                training_states["losses"].append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ts.write(
                    "add_scalar",
                    tag="Loss/training_loss",
                    scalar_value=training_states["losses"][-1],
                    global_step=training_states["global_step"],
                )
                training_states["global_step"] += 1
                if rank == 0:
                    print(f"Epoch: {epoch}, Step: {i}, Loss: {loss}")

            ts.save(epoch, model.module, optimizer, None, training_states)

            if rank == 0:
                model.eval()
                with torch.no_grad():              
                    ori_img = torch.normal(0, 1, (3, args.img_size, args.img_size)).cuda()
                    mixed_img = ori_img.unsqueeze(0)
                    for step in range(1000, 0, -1):
                        step = torch.tensor([step])
                        pred_noise = model(mixed_img, step)
                        signal_rate, noise_rate = linear_schedule(step-1)
                        signal_rate, noise_rate = signal_rate.cuda(), noise_rate.cuda()
                        pred_img = (mixed_img - noise_rate * pred_noise) / signal_rate
                        next_signal_rate, next_noise_rate = linear_schedule(step-2)
                        next_signal_rate, next_noise_rate = next_signal_rate.cuda(), next_noise_rate.cuda()
                        mixed_img = next_signal_rate * pred_img + next_noise_rate * pred_noise
                    pred_img = torch.permute(pred_img.cpu()[0], (1, 2, 0))
                    pred_img = (pred_img * STD + MEAN) * 255
                    pred_img = pred_img.clip(min=0, max=255).type(torch.uint8)
                    ts.write(
                            "add_image",
                            tag="Image/samples",
                            img_tensor=pred_img,
                            global_step=training_states["global_step"],
                            dataformats='HWC'
                    )
                model.train()


def main():
    args = init_argparser()
    mp.spawn(
        train,
        args=[args],
        nprocs=args.num_gpu,
        join=True,
    )


if __name__ == '''__main__''':
    main()
