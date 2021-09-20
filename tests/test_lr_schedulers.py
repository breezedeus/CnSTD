import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from cnstd.lr_scheduler import WarmupCosineAnnealingRestarts


class NullModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)


ori_lr = 5e-4
model = NullModule()
optimizer = torch.optim.Adam(model.parameters())


def plot_lr(scheduler, step=900):
    lrs = []
    for i in range(step):
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        lrs.append(lr)

    plt.plot(lrs)
    plt.show()


def test_CosineAnnealingWarmRestarts():
    CAW = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=1, eta_min=ori_lr / 10.0
    )
    plot_lr(CAW, step=1000)


def test_WarmupCosineAnnealingRestarts():
    CAW = WarmupCosineAnnealingRestarts(
        optimizer,
        first_cycle_steps=95600,
        cycle_mult=1.0,
        max_lr=0.001,
        min_lr=0.0001,
        warmup_steps=100,
        gamma=1.0,
    )
    plot_lr(CAW, step=95600)


def test_CyclicLR():
    Cyc = lr_scheduler.CyclicLR(
        optimizer,
        base_lr=ori_lr / 10.0,
        max_lr=ori_lr,
        step_size_up=200,
        cycle_momentum=False,
    )

    plot_lr(Cyc, 1000)


def test_OneCycleLR():
    Cyc = lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, epochs=20, steps_per_epoch=50,
    )

    plot_lr(Cyc, 1000)
