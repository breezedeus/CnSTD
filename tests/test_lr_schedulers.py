import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt


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
        optimizer, T_0=8, T_mult=1, eta_min=ori_lr / 10.0
    )
    plot_lr(CAW, step=40)


def test_CyclicLR():
    Cyc = lr_scheduler.CyclicLR(optimizer,
        base_lr=ori_lr / 10.0, max_lr=ori_lr, step_size_up=5, cycle_momentum=False,
    )

    plot_lr(Cyc, 50)
