import os
import re
import time

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(model, optimizer, scheduler, epoch, trainin_states, save_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "trainin_states": trainin_states,
        },
        save_path,
    )


def load_checkpoint(save_path):
    checkpoint = torch.load(save_path, map_location="cpu")

    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    scheduler_state_dict = checkpoint["scheduler_state_dict"]
    epoch = checkpoint["epoch"]
    trainin_states = checkpoint["trainin_states"]
    return model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, trainin_states


class MultiGPUTraningState:
    def __init__(
        self,
        rank,
        sz_world,
        model_path,
        prefix,
        log_dir,
        max_saving_num=4,
        seed=10,
        master_addr="localhost",
        master_port="19931",
    ):
        torch.cuda.set_device(rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        dist.init_process_group("nccl", rank=rank, world_size=sz_world)
        torch.manual_seed(seed)

        self.rank = rank
        self.sz_world = sz_world
        self.model_path = model_path
        self.prefix = prefix
        self.max_saving_num = max_saving_num
        self.id = 0
        self.init_epoch = -1  # 注意，epoch从0开始，对应的状态备份也是
        self.training_states = None  # 用于记录训练时的状态(如历史loss)
        self.model_ids = []
        self.model_state_dict = None
        self.optim_state_dict = None
        self.sched_state_dict = None
        self._writer = None

        if rank == 0:
            log_path = log_dir + "/training_log_{id}"

            log_id = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
            while os.path.exists(log_path.format(id=log_id)):
                log_id = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
            log_path = log_path.format(id=log_id)
            print("log path: %s" % log_path)

            self._writer = SummaryWriter(log_path)

    def __enter__(self):
        for _, _, fns in os.walk(self.model_path):
            for fn in fns:
                id = re.findall(self.prefix + r"(\d+)", fn)
                if len(id) > 0:
                    self.model_ids.append(int(id[0]))
        self.model_ids.sort(reverse=True)

        if len(self.model_ids) > 0:
            (
                self.model_state_dict,
                self.optim_state_dict,
                self.sched_state_dict,
                self.init_epoch,
                self.training_states,
            ) = load_checkpoint(self.model_path + "/" + self.prefix + "%d" % self.model_ids[0])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.destroy_process_group()

    def save(self, epoch, model, optimizer, scheduler, training_states):
        if self.rank == 0:
            self.model_ids.insert(0, epoch)
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                training_states,
                self.model_path + "/" + self.prefix + "%d" % self.model_ids[0],
            )
            while len(self.model_ids) > self.max_saving_num:
                rm_fn = self.model_path + "/" + self.prefix + "%d" % self.model_ids[-1]
                if os.path.exists(rm_fn):
                    os.remove(rm_fn)
                del self.model_ids[-1]

    def write(self, fn, **kwargs):
        if self.rank == 0:
            getattr(self._writer, fn)(**kwargs)
