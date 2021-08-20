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
import json
import time

import numpy as np
import torchvision.transforms as T

from .consts import BACKBONE_NET_NAME, MODEL_VERSION
from .utils import (
    set_logger,
    data_dir,
    load_model_params,
    imsave,
    read_img,
)
from .dataset import StdDataModule
from .trainer import PlTrainer, resave_model
from .model import gen_model
from .model.core import DetectionPredictor

# from .eval import evaluate


_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}

logger = set_logger(log_level='INFO')
DEFAULT_MODEL_NAME = 'db_resnet18'


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train')
@click.option('-m', '--model-name', type=str, default=DEFAULT_MODEL_NAME, help='模型名称')
@click.option(
    '-i',
    '--index-dir',
    type=str,
    required=True,
    help='索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件',
)
@click.option('--train-config-fp', type=str, required=True, help='训练使用的json配置文件')
@click.option(
    '-r', '--resume-from-checkpoint', type=str, default=None, help='恢复此前中断的训练状态，继续训练'
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='导入的训练好的模型，作为初始模型。优先级低于"--restore-training-fp"，当传入"--restore-training-fp"时，此传入可能失效',
)
def train(
    model_name, index_dir, train_config_fp, resume_from_checkpoint, pretrained_model_fp
):
    train_config = json.load(open(train_config_fp))

    kwargs = dict(rotated_bbox=train_config['rotated_bbox'])
    if 'resized_shape' in train_config:
        kwargs['input_shape'] = train_config['resized_shape']
    model = gen_model(model_name, **kwargs)
    logger.info(model)
    logger.info(model.cfg)
    expected_img_shape = model.cfg['input_shape']

    train_transform = T.Compose(  # MUST NOT include `Resize`
        [
            T.RandomInvert(p=0.2),
            T.RandomPosterize(bits=4, p=0.3),
            T.RandomAdjustSharpness(sharpness_factor=0.5, p=0.3),
            T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2),
            T.RandomEqualize(p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
        ]
    )
    val_transform = None

    data_mod = StdDataModule(
        index_dir=index_dir,
        resized_shape=expected_img_shape[1:],
        preserve_aspect_ratio=train_config['preserve_aspect_ratio'],
        data_root_dir=train_config['data_root_dir'],
        train_transforms=train_transform,
        val_transforms=val_transform,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
        debug=train_config.get('debug', False),
    )

    # train_ds = data_mod.train
    # visualize_example(train_ds[8])
    # return

    trainer = PlTrainer(
        train_config, ckpt_fn=['cnstd', 'v%s' % MODEL_VERSION, model_name]
    )

    if pretrained_model_fp is not None:
        load_model_params(model, pretrained_model_fp)

    trainer.fit(
        model, datamodule=data_mod, resume_from_checkpoint=resume_from_checkpoint
    )


def visualize_example(example):
    image = example['image']
    imsave(image, 'debug-image.jpg', normalized=True)

    def _vis_bool(img, fp):
        img *= 255
        imsave(img, fp, normalized=False)

    _vis_bool(example['gt'].transpose(1, 2, 0), 'debug-gt.jpg')
    _vis_bool(np.expand_dims(example['mask'], -1), 'debug-mask.jpg')
    _vis_bool(np.expand_dims(example['thresh_map'], -1), 'debug-thresh-map.jpg')
    _vis_bool(np.expand_dims(example['thresh_mask'], -1), 'debug-thresh-mask.jpg')


@cli.command('predict')
@click.option('-m', '--model-name', type=str, default=DEFAULT_MODEL_NAME, help='模型名称')
@click.option("--model-epoch", type=int, default=None, help="model epoch")
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='导入的训练好的模型，作为初始模型。优先级低于"--restore-training-fp"，当传入"--restore-training-fp"时，此传入可能失效',
)
@click.option("-r", "--rotated-bbox", is_flag=True, help="是否考虑旋转box")
@click.option(
    "--resized-shape",
    type=str,
    default='768,768',
    help='格式："height,width"; 预测时把图片resize到此大小再进行预测。两个值都需要是32的倍数',
)
@click.option("--box-score-thresh", type=float, default=0.3, help="只考虑分数大于此值的boxes")
@click.option(
    "--preserve-aspect-ratio", type=bool, default=True, help="resize时是否保留图片原始比例"
)
@click.option(
    "--context",
    help="使用cpu还是gpu运行代码。默认为cpu",
    type=click.Choice(['cpu', 'gpu']),
    default='cpu',
)
@click.option("-f", "--file", help="Path to the image file or dir")
def predict(
    model_name,
    model_epoch,
    pretrained_model_fp,
    rotated_bbox,
    resized_shape,
    box_score_thresh,
    preserve_aspect_ratio,
    context,
    file,
):
    model = gen_model(model_name, pretrained_backbone=False, rotated_bbox=rotated_bbox)
    model.eval()
    if pretrained_model_fp is not None:
        load_model_params(model, pretrained_model_fp)

    resized_shape = tuple(map(int, resized_shape.split(',')))  # [H, W]
    predictor = DetectionPredictor(
        model,
        resized_shape=resized_shape,
        preserve_aspect_ratio=preserve_aspect_ratio,
        debug=True,
    )
    pil_img = read_img(file)
    start_time = time.time()
    predictor([pil_img], box_score_thresh=box_score_thresh)
    logger.info('time cost of prediction: %f' % (time.time() - start_time))


@cli.command('resave')
@click.option('-i', '--input-model-fp', type=str, required=True, help='输入的模型文件路径')
@click.option('-o', '--output-model-fp', type=str, required=True, help='输出的模型文件路径')
def resave_model_file(
        input_model_fp,
        output_model_fp,
):
    resave_model(input_model_fp, output_model_fp, map_location='cpu')


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
@click.option('-o', '--output_dir', default='outputs', help='输出结果存放的目录')
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
