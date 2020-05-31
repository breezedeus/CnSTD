# coding: utf-8
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
import os
import click

from .consts import BACKBONE_NET_NAME, MODEL_VERSION
from .utils import set_logger, gen_context, data_dir
from .train import train
from .eval import evaluate


_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}

logger = set_logger(log_level='DEBUG')


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train', context_settings=_CONTEXT_SETTINGS)
@click.option(
    '--backbone',
    type=click.Choice(BACKBONE_NET_NAME),
    default='mobilenetv3',
    help='backbone model name',
)
@click.option('--pretrain_model_fp', type=str, default=None, help='初始化模型路径')
@click.option('--gpu', type=int, default=-1, help='使用的GPU数量。默认值为-1，表示自动判断')
@click.option(
    "--optimizer",
    type=str,
    default='Adam',
    help="optimizer for training [Default: Adam]",
)
@click.option(
    '--batch_size', type=int, default=4, help='batch size for each device [Default: 4]'
)
@click.option('--epoch', type=int, default=50, help='train epochs [Default: 50]')
@click.option('--lr', type=float, default=0.001, help='learning rate [Default: 0.001]')
@click.option('--momentum', type=float, default=0.99, help='momentum [Default: 0.9]')
@click.option(
    '--wd', type=float, default=5e-4, help='weight decay factor [Default: 0.0]'
)
@click.option('--log_step', type=int, default=5, help='隔多少步打印一次信息 [Default: 5]')
@click.option('-r', '--root_dir', type=str, help='数据所在的根目录，它与索引文件中指定的文件路径合并后获得最终的文件路径')
@click.option('-i', '--train_index_fp', type=str, help='存放训练数据的索引文件')
@click.option('-o', '--output_dir', default=data_dir(), help='模型输出的根目录')
def train_model(
    backbone,
    pretrain_model_fp,
    gpu,
    optimizer,
    batch_size,
    epoch,
    lr,
    momentum,
    wd,
    log_step,
    root_dir,
    train_index_fp,
    output_dir,
):
    devices = gen_context(gpu)
    output_dir = os.path.join(output_dir, MODEL_VERSION)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(
        backbone=backbone,
        root_dir=root_dir,
        train_index_fp=train_index_fp,
        pretrain_model=pretrain_model_fp,
        optimizer=optimizer,
        batch_size=batch_size,
        epochs=epoch,
        lr=lr,
        momentum=momentum,
        wd=wd,
        verbose_step=log_step,
        output_dir=output_dir,
        ctx=devices,
    )


@cli.command('evaluate', context_settings=_CONTEXT_SETTINGS)
@click.option(
    '--backbone',
    type=click.Choice(BACKBONE_NET_NAME),
    default='mobilenetv3',
    help='backbone model name',
)
@click.option('--model_root_dir', default=data_dir(), help='模型所在的根目录')
@click.option('--model_epoch', type=int, default=None, help='model epoch')
@click.option('-i', '--img_dir', type=str, help='评估图片所在的目录或者单个图片文件路径')
@click.option(
    '--max_size',
    type=int,
    default=768,
    help='图片预测时的最大尺寸（最好是32的倍数）。超过这个尺寸的图片会被等比例压缩到此尺寸 [Default: 768]',
)
@click.option(
    '--pse_threshold',
    type=float,
    default=0.45,
    help='threshold for pse [Default: 0.45]',
)
@click.option(
    '--pse_min_area', type=int, default=100, help='min area for pse [Default: 100]'
)
@click.option('--gpu', type=int, default=-1, help='使用的GPU数量。默认值为-1，表示自动判断')
@click.option('-o', '--output_dir', default='outputs', help='模型输出的目录')
def evaluate_model(
    backbone,
    model_root_dir,
    model_epoch,
    img_dir,
    max_size,
    pse_threshold,
    pse_min_area,
    gpu,
    output_dir,
):
    devices = gen_context(gpu)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    evaluate(
        backbone,
        model_root_dir,
        model_epoch,
        img_dir,
        output_dir,
        max_size,
        pse_threshold,
        pse_min_area,
        devices,
    )


if __name__ == '__main__':
    cli()
