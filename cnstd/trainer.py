# coding: utf-8
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union, List

import numpy as np
import torch
import torch.optim as optim
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import (
    StepLR,
    LambdaLR,
    CyclicLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
)
from torch.utils.data import DataLoader

from .utils import LocalizationConfusion

logger = logging.getLogger(__name__)


def get_optimizer(name: str, model, learning_rate, weight_decay):
    r"""Init the Optimizer

    Returns:
        torch.optim: the optimizer
    """
    OPTIMIZERS = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'adagrad': optim.Adagrad,
        'rmsprop': optim.RMSprop,
    }

    try:
        opt_cls = OPTIMIZERS[name.lower()]
        optimizer = opt_cls(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    except:
        logger.warning('Received unrecognized optimizer, set default Adam optimizer')
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    return optimizer


def get_lr_scheduler(config, optimizer):
    orig_lr = config['learning_rate']
    lr_sch_config = deepcopy(config['lr_scheduler'])
    lr_sch_name = lr_sch_config.pop('name')

    if lr_sch_name == 'multi_step':
        return MultiStepLR(
            optimizer,
            milestones=lr_sch_config['milestones'],
            gamma=lr_sch_config['gamma'],
        )
    elif lr_sch_name == 'cos_anneal':
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=4, T_mult=1, eta_min=orig_lr / 10.0
        )
    elif lr_sch_name == 'cyclic':
        return CyclicLR(
            optimizer,
            base_lr=orig_lr / 10.0,
            max_lr=orig_lr,
            step_size_up=2,
            cycle_momentum=False,
        )

    step_size = lr_sch_config['step_size']
    gamma = lr_sch_config['gamma']
    if step_size is None or gamma is None:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1)
    return StepLR(optimizer, step_size, gamma=gamma)


class WrapperLightningModule(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self._optimizer = get_optimizer(
            config['optimizer'],
            self.model,
            config['learning_rate'],
            config.get('weight_decay', 0),
        )

        expected_img_shape = model.cfg['input_shape']
        self.val_metric = LocalizationConfusion(
            rotated_bbox=self.model.rotated_bbox, mask_shape=expected_img_shape[1:]
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if hasattr(self.model, 'set_current_epoch'):
            self.model.set_current_epoch(self.current_epoch)
        else:
            setattr(self.model, 'current_epoch', self.current_epoch)
        res = self.model.calculate_loss(batch)
        losses = res['loss']
        self.log(
            'train_loss',
            losses.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return losses

    def validation_step(self, batch, batch_idx):
        if hasattr(self.model, 'validation_step'):
            return self.model.validation_step(batch, batch_idx, self)

        res = self.model.calculate_loss(
            batch, return_model_output=True, return_preds=True
        )
        losses = res['loss']
        val_metrics = {'val_loss': losses.item()}
        self.log_dict(
            val_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )

        pred_boxes = [boxes[:, :-1] for boxes in res['preds'][0]]  # 最后一列是分数，去掉不用
        match, mean_iou = self.val_metric.update(
            batch['polygons'], pred_boxes
        )
        val_metrics = dict(match_step=match, mean_iou_step=mean_iou)
        self.log_dict(
            val_metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )

        return losses

    def validation_epoch_end(self, losses_list) -> None:
        match, mean_iou = self.val_metric.summary()
        val_metrics = dict(match_epoch=match, mean_iou_epoch=mean_iou)
        self.log_dict(
            val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        self.val_metric.reset()

    def configure_optimizers(self):
        return [self._optimizer], [get_lr_scheduler(self.config, self._optimizer)]


class PlTrainer(object):
    """
    封装 PyTorch Lightning 的训练器。
    """

    def __init__(self, config, ckpt_fn=None):
        self.config = config

        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor]

        mode = self.config.get('pl_checkpoint_mode', 'min')
        monitor = self.config.get('pl_checkpoint_monitor')
        fn_fields = ckpt_fn or []
        fn_fields.append('{epoch:03d}')
        if monitor:
            fn_fields.append('{' + monitor + ':.4f}')
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                filename='-'.join(fn_fields),
                save_last=True,
                save_top_k=5,
            )
            callbacks.append(checkpoint_callback)

        self.pl_trainer = pl.Trainer(
            limit_train_batches=self.config.get('limit_train_batches', 1.0),
            limit_val_batches=self.config.get('limit_val_batches', 1.0),
            num_sanity_val_steps=2,
            gpus=self.config.get('gpus'),
            max_epochs=self.config.get('epochs', 20),
            precision=self.config.get('precision', 32),
            callbacks=callbacks,
            stochastic_weight_avg=True,
        )

    def fit(
        self,
        model: nn.Module,
        train_dataloader: Any = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
    ):
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloader: Either a single PyTorch DataLoader or a collection of these
                (list, dict, nested lists and dicts). In the case of multiple dataloaders, please
                see this :ref:`page <multiple-training-dataloaders>`
            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped
            datamodule: A instance of :class:`LightningDataModule`.
            resume_from_checkpoint: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, start from scratch. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.
        """
        if resume_from_checkpoint is not None:
            pl_module = WrapperLightningModule.load_from_checkpoint(
                resume_from_checkpoint, config=self.config, model=model
            )
            self.pl_trainer = pl.Trainer(resume_from_checkpoint=resume_from_checkpoint)
        else:
            pl_module = WrapperLightningModule(self.config, model)

        self.pl_trainer.fit(pl_module, train_dataloader, val_dataloaders, datamodule)

        fields = self.pl_trainer.checkpoint_callback.best_model_path.rsplit(
            '.', maxsplit=1
        )
        fields[0] += '-model'
        output_model_fp = '.'.join(fields)
        resave_model(
            self.pl_trainer.checkpoint_callback.best_model_path, output_model_fp
        )
        self.saved_model_file = output_model_fp


def resave_model(module_fp, output_model_fp):
    """PlTrainer存储的文件对应其 `pl_module` 模块，需利用此函数转存为 `model` 对应的模型文件。"""
    checkpoint = torch.load(module_fp)
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k.split('.', maxsplit=1)[1]] = v
    torch.save({'state_dict': state_dict}, output_model_fp)
