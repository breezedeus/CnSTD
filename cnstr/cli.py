# coding=utf-8
import os
import click

from .utils import set_logger, gen_context
from .train import train
from .eval import evaluate


_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}

logger = set_logger('INFO')


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train', context_settings=_CONTEXT_SETTINGS)
@click.option('-i', '--data_dir', type=str, help='数据所在的根目录')
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
@click.option('-o', '--output_dir', default='ckpt', help='模型输出的目录')
def train_model(
    data_dir,
    pretrain_model_fp,
    gpu,
    optimizer,
    batch_size,
    epoch,
    lr,
    momentum,
    wd,
    log_step,
    output_dir,
):
    devices = gen_context(gpu)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(
        data_dir=data_dir,
        pretrain_model=pretrain_model_fp,
        optimizer=optimizer,
        batch_size=batch_size,
        epochs=epoch,
        lr=lr,
        momentum=momentum,
        wd=wd,
        verbose_step=log_step,
        ckpt=output_dir,
        ctx=devices,
    )


@cli.command('evaluate', context_settings=_CONTEXT_SETTINGS)
@click.option('-i', '--data_dir', type=str, help='数据所在的根目录')
@click.option('--model_fp', type=str, default=None, help='模型路径')
@click.option('--gpu', type=int, default=-1, help='使用的GPU数量。默认值为-1，表示自动判断')
@click.option(
    '--batch_size', type=int, default=4, help='batch size for each device [Default: 4]'
)
@click.option('-o', '--output_dir', default='ckpt', help='模型输出的目录')
def evaluate_model(data_dir, model_fp, gpu, batch_size, output_dir):
    devices = gen_context(gpu)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    evaluate(data_dir, model_fp, output_dir, devices)


if __name__ == '__main__':
    cli()
