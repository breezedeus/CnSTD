# coding=utf-8
import os
import click

from .utils import set_logger, gen_context
from .train import train
from .eval import evaluate


_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}

logger = set_logger(log_level='INFO')


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train', context_settings=_CONTEXT_SETTINGS)
@click.option(
    '--backbone',
    type=click.Choice(['mobilenetv3', 'resnet50_v1b']),
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
@click.option('-o', '--output_dir', default='ckpt', help='模型输出的目录')
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
@click.option('-i', '--data_dir', type=str, help='数据所在的根目录')
@click.option('--model_fp', type=str, default=None, help='模型路径')
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
@click.option(
    '--batch_size', type=int, default=4, help='batch size for each device [Default: 4]'
)
@click.option('-o', '--output_dir', default='outputs', help='模型输出的目录')
def evaluate_model(
    data_dir,
    model_fp,
    max_size,
    pse_threshold,
    pse_min_area,
    gpu,
    batch_size,
    output_dir,
):
    devices = gen_context(gpu)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    evaluate(
        data_dir, model_fp, output_dir, max_size, pse_threshold, pse_min_area, devices
    )


if __name__ == '__main__':
    cli()
