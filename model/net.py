# coding=utf-8
from mxnet.gluon import HybridBlock, nn
import mxnet as mx
from gluoncv import model_zoo
from gluoncv.model_zoo.resnetv1b import resnet18_v1b
from .feature import FPNFeatureExpander
from mxnet.gluon.contrib.nn import SyncBatchNorm


class PSENet(HybridBlock):
    def __init__(
        self,
        num_kernels,
        scale=1,
        ctx=mx.cpu(),
        pretrained=False,
        num_device=0,
        **kwargs
    ):
        super(PSENet, self).__init__()
        self.num_kernels = num_kernels

        base_network = resnet18_v1b(
            pretrained=pretrained,
            dilated=False,
            use_global_stats=False,
            norm_layer=nn.BatchNorm,
            ctx=ctx,
            **kwargs
        )
        self.features = FPNFeatureExpander(
            network=base_network,
            outputs=[
                'layers1_relu8_fwd',
                'layers2_relu11_fwd',
                'layers3_relu17_fwd',
                'layers4_relu8_fwd',
            ],
            num_filters=[256, 256, 256, 256],
            use_1x1=True,
            use_upsample=True,
            use_elewadd=True,
            use_p6=False,
            no_bias=True,
            pretrained=pretrained,
            ctx=ctx,
        )

        self.scale = scale
        self.extrac_convs = []

        for i in range(4):
            weight_init = mx.init.Normal(0.001)
            extra_conv = nn.HybridSequential(prefix='extra_conv_{}'.format(i))
            with extra_conv.name_scope():
                extra_conv.add(nn.Conv2D(256, 3, 1, 1))
                # extra_conv.add(nn.BatchNorm())
                extra_conv.add(nn.Activation('relu'))
            extra_conv.initialize(weight_init, ctx=ctx)
            self.register_child(extra_conv)
            self.extrac_convs.append(extra_conv)

        self.decoder_out = nn.HybridSequential(prefix='decoder_out')
        with self.decoder_out.name_scope():
            weight_init = mx.init.Normal(0.001)
            self.decoder_out.add(nn.Conv2D(256, 3, 1, 1))
            # self.decoder_out.add(nn.BatchNorm())
            self.decoder_out.add(nn.Activation('relu'))
            self.decoder_out.add(nn.Conv2D(self.num_kernels, 1, 1))
            self.decoder_out.initialize(weight_init, ctx=ctx)

    def hybrid_forward(self, F, x, **kwargs):
        # output: c4 -> c1 [1/4, 1/8, 1/16. 1/32]
        fpn_features = self.features(x)

        concat_features = []
        scales = [1, 2, 4, 8]
        for i, C in enumerate(fpn_features):
            extrac_C = self.extrac_convs[i](C)
            up_C = F.UpSampling(
                extrac_C,
                scale=scales[i],
                sample_type='nearest',
                name="extra_upsample_{}".format(i),
            )
            concat_features.append(up_C)
        concat_output = F.concat(*concat_features, dim=1)
        output = self.decoder_out(concat_output)
        if self.scale > 1.0:
            output = F.UpSampling(
                output, scale=self.scale, sample_type='nearest', name="final_upsampling"
            )
        output = F.sigmoid(output)
        return output


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    fpn = PSENet(num_kernels=7, pretrained=True)
    fpn.initialize(ctx=mx.cpu())
    x = mx.nd.array([np.random.uniform(-2, 4.2, size=(3, 640, 640))])
    x = fpn(x)
    # fpn.hybridize()
    # x = x.asnumpy()
    # print x.shape
    score_map = x[0, 6, :, :].asnumpy()
    kernel_map = x[0, 0, :, :].asnumpy()

    plt.imshow(score_map)
    plt.show()

    plt.imshow(kernel_map)
    plt.show()
