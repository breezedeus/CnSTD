# coding=utf-8
from mxnet.gluon import trainer
from datasets.dataloader import ICDAR
from mxnet.gluon.data import DataLoader
from model.net import PSENet
import mxnet as mx
from mxnet.gluon import Trainer
from model.loss import DiceLoss, DiceLoss_with_OHEM
from mxnet import autograd

from mxnet import lr_scheduler as ls
import os
from tensorboardX import SummaryWriter


def train(
    data_dir,
    pretrain_model,
    epoches=50,
    lr=0.001,
    wd=5e-4,
    momentum=0.9,
    batch_size=5,
    ctx=mx.cpu(),
    verbose_step=1,
    ckpt='ckpt',
):
    num_kernels = 3
    icdar_loader = ICDAR(root_dir=data_dir, num_kernels=num_kernels - 1)
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
    pse_loss = DiceLoss_with_OHEM(lam=0.7, num_kernels=num_kernels)

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

        for i, item in enumerate(loader):
            im, score_maps, kernels, training_masks, ori_img = item

            im = im.as_in_context(ctx)
            score_maps = score_maps[:, ::4, ::4].as_in_context(ctx)
            kernels = kernels[:, :, ::4, ::4].as_in_context(ctx)
            training_masks = training_masks[:, ::4, ::4].as_in_context(ctx)

            with autograd.record():
                kernels_pred = net(im)
                loss = pse_loss(score_maps, kernels, kernels_pred, training_masks)
                loss.backward()
            trainer.step(batch_size)
            if i % verbose_step == 0:
                global_steps = icdar_loader.length * e + i * batch_size
                summary_writer.add_image(
                    'score_map', score_maps[0:1, :, :], global_steps
                )
                summary_writer.add_image(
                    'score_map_pred', kernels_pred[0:1, -1, :, :], global_steps
                )
                summary_writer.add_image(
                    'kernel_map', kernels[0:1, 0, :, :], global_steps
                )
                summary_writer.add_image(
                    'kernel_map_pred', kernels_pred[0:1, 0, :, :], global_steps
                )
                summary_writer.add_scalar(
                    'loss', mx.nd.mean(loss).asscalar(), global_steps
                )
                summary_writer.add_scalar(
                    'c_loss', mx.nd.mean(pse_loss.C_loss).asscalar(), global_steps
                )
                summary_writer.add_scalar(
                    'kernel_loss',
                    mx.nd.mean(pse_loss.kernel_loss).asscalar(),
                    global_steps,
                )
                summary_writer.add_scalar(
                    'pixel_accuracy', pse_loss.pixel_acc, global_steps
                )
            if i % 1 == 0:
                print(
                    "step: {}, loss: {}, score_loss: {}, kernel_loss: {}, pixel_acc: {}, kernel_acc:{}".format(
                        i * batch_size,
                        mx.nd.mean(loss).asscalar(),
                        mx.nd.mean(pse_loss.C_loss).asscalar(),
                        mx.nd.mean(pse_loss.kernel_loss).asscalar(),
                        pse_loss.pixel_acc,
                        pse_loss.kernel_acc,
                    )
                )
            cumulative_loss += mx.nd.mean(loss).asscalar()
        print("Epoch {}, loss: {}".format(e, cumulative_loss))
        net.save_parameters(os.path.join(ckpt, 'model_{}.param'.format(e)))
    summary_writer.close()


if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1]
    pretrain_model = sys.argv[2]
    ctx = sys.argv[3]
    if len(sys.argv) < 2:
        print("Usage: python train.py $data_dir $pretrain_model")
    if eval(ctx) >= 0:
        devices = mx.gpu(eval(ctx))
    else:
        devices = mx.cpu()
    train(data_dir=data_dir, pretrain_model=pretrain_model, ctx=devices)
