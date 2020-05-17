# coding=utf-8
import os
import numpy as np
import mxnet as mx
from mxnet.gluon.data import DataLoader
from mxnet.gluon import Trainer
from mxnet import autograd, gluon, lr_scheduler as ls
from tensorboardX import SummaryWriter

from datasets.dataloader import ICDAR
from model.loss import DiceLoss, DiceLoss_with_OHEM
from model.net import PSENet


def train(
    data_dir,
    pretrain_model,
    epoches=50,
    lr=0.001,
    wd=5e-4,
    momentum=0.9,
    batch_size=4,
    ctx=mx.cpu(),
    verbose_step=5,
    ckpt='ckpt',
):
    num_kernels = 3
    icdar_loader = ICDAR(root_dir=data_dir, num_kernels=num_kernels - 1)
    if not isinstance(ctx, (list, tuple)):
        ctx = [ctx]
    batch_size = batch_size * len(ctx)
    loader = DataLoader(icdar_loader, batch_size=batch_size, shuffle=True)
    net = PSENet(num_kernels=num_kernels, ctx=ctx, pretrained=True)
    # initial params
    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    net.collect_params("extra_*_weight | decoder_*_weight").initialize(
        mx.init.Xavier(), ctx=ctx, force_reinit=True
    )
    net.collect_params("extra_*_bias | decoder_*_bias").initialize(
        mx.init.Zero(), ctx=ctx, force_reinit=True
    )
    net.collect_params("!(resnet*)").setattr("lr_mult", 10)
    net.collect_params("!(resnet*)").setattr('grad_req', 'null')
    net.load_parameters(pretrain_model, ctx=ctx, allow_missing=True, ignore_extra=True)
    # net.initialize(ctx=ctx)

    # pse_loss = DiceLoss(lam=0.7, num_kernels=num_kernels)
    pse_loss = DiceLoss_with_OHEM(lam=0.7, num_kernels=num_kernels, debug=False)

    cos_shc = ls.PolyScheduler(
        max_update=icdar_loader.length * epoches // batch_size, base_lr=lr
    )
    trainer = Trainer(
        net.collect_params(),
        'sgd',
        {'learning_rate': lr, 'wd': wd, 'momentum': momentum, 'lr_scheduler': cos_shc},
    )
    summary_writer = SummaryWriter(ckpt)
    for e in range(epoches):
        cumulative_loss = 0

        num_batches = 0
        for i, item in enumerate(loader):
            item_ctxs = [split_and_load(field, ctx) for field in item]
            # item_ctxs = split_and_load(item, ctx)
            loss_list = []
            for im, score_maps, kernels, training_masks, ori_img in zip(*item_ctxs):
                # im, score_maps, kernels, training_masks, ori_img = item
                # im = im.as_in_context(ctx)
                # import pdb; pdb.set_trace()
                score_maps = score_maps[:, ::4, ::4]
                kernels = kernels[:, :, ::4, ::4]
                training_masks = training_masks[:, ::4, ::4]

                with autograd.record():
                    kernels_pred = net(im)
                    loss = pse_loss(score_maps, kernels, kernels_pred, training_masks)
                    loss_list.append(loss)
            mean_loss = []
            for loss in loss_list:
                loss.backward()
                mean_loss.append(mx.nd.mean(to_cpu(loss)).asscalar())
            mean_loss = np.mean(mean_loss)
            trainer.step(batch_size)
            if i % verbose_step == 0:
                global_steps = icdar_loader.length * e + i * batch_size
                summary_writer.add_image(
                    'score_map', to_cpu(score_maps[0:1, :, :]), global_steps
                )
                summary_writer.add_image(
                    'score_map_pred', to_cpu(kernels_pred[0:1, -1, :, :]), global_steps
                )
                summary_writer.add_image(
                    'kernel_map', to_cpu(kernels[0:1, 0, :, :]), global_steps
                )
                summary_writer.add_image(
                    'kernel_map_pred', to_cpu(kernels_pred[0:1, 0, :, :]), global_steps
                )
                summary_writer.add_scalar('loss', mean_loss, global_steps)
                summary_writer.add_scalar(
                    'c_loss', mx.nd.mean(to_cpu(pse_loss.C_loss)).asscalar(), global_steps
                )
                summary_writer.add_scalar(
                    'kernel_loss',
                    mx.nd.mean(to_cpu(pse_loss.kernel_loss)).asscalar(),
                    global_steps,
                )
                summary_writer.add_scalar(
                    'pixel_accuracy', pse_loss.pixel_acc, global_steps
                )
            if i % 1 == 0:
                print(
                    "step: {}, loss: {}, score_loss: {}, kernel_loss: {}, pixel_acc: {}, kernel_acc:{}".format(
                        i * batch_size,
                        mean_loss,
                        mx.nd.mean(to_cpu(pse_loss.C_loss)).asscalar(),
                        mx.nd.mean(to_cpu(pse_loss.kernel_loss)).asscalar(),
                        pse_loss.pixel_acc,
                        pse_loss.kernel_acc,
                    )
                )
            cumulative_loss += mean_loss
            num_batches += 1
        print("Epoch {}, mean loss: {}".format(e, cumulative_loss / num_batches))
        net.save_parameters(os.path.join(ckpt, 'model_{}.param'.format(e)))
    summary_writer.close()


def to_cpu(nd_array):
    return nd_array.as_in_context(mx.cpu())


def split_and_load(xs, ctx_list):
    return gluon.utils.split_and_load(
        xs, ctx_list=ctx_list, batch_axis=0, even_split=False
    )


if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1]
    pretrain_model = sys.argv[2]
    num_gpus = int(sys.argv[3])
    if len(sys.argv) < 3:
        print("Usage: python train.py $data_dir $pretrain_model $num_gpus")
    if num_gpus > 0:
        devices = [mx.gpu(i) for i in range(num_gpus)]
    else:
        devices = [mx.cpu()]
    train(data_dir=data_dir, pretrain_model=pretrain_model, ctx=devices)
