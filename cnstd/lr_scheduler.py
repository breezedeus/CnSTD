# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from copy import deepcopy
import math

import torch
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    LambdaLR,
    CyclicLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    OneCycleLR,
)


def get_lr_scheduler(config, optimizer):
    orig_lr = config['learning_rate']
    lr_sch_config = deepcopy(config['lr_scheduler'])
    lr_sch_name = lr_sch_config.pop('name')
    epochs = config['epochs']
    steps_per_epoch = config['steps_per_epoch']

    if lr_sch_name == 'multi_step':
        milestones = [v * steps_per_epoch for v in lr_sch_config['milestones']]
        return MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_sch_config['gamma'],
        )
    elif lr_sch_name == 'cos_warmup':
        min_lr_mult_factor = lr_sch_config.get('min_lr_mult_factor', 0.1)
        warmup_epochs = lr_sch_config.get('warmup_epochs', 0.1)
        return WarmupCosineAnnealingRestarts(
            optimizer,
            first_cycle_steps=steps_per_epoch * epochs,
            max_lr=orig_lr,
            min_lr=orig_lr * min_lr_mult_factor,
            warmup_steps=int(steps_per_epoch * warmup_epochs),
        )
    elif lr_sch_name == 'cos_anneal':
        # 5 个 epochs, 一个循环
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=5 * steps_per_epoch, T_mult=1, eta_min=orig_lr * 0.1
        )
    elif lr_sch_name == 'cyclic':
        return CyclicLR(
            optimizer,
            base_lr=orig_lr / 10.0,
            max_lr=orig_lr,
            step_size_up=5 * steps_per_epoch,  # 5 个 epochs, 从最小base_lr上升到最大max_lr
            cycle_momentum=False,
        )
    elif lr_sch_name == 'one_cycle':
        return OneCycleLR(
            optimizer, max_lr=orig_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        )

    step_size = lr_sch_config['step_size']
    gamma = lr_sch_config['gamma']
    if step_size is None or gamma is None:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1)
    return StepLR(optimizer, step_size, gamma=gamma)


class WarmupCosineAnnealingRestarts(_LRScheduler):
    """
    from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult ** n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
