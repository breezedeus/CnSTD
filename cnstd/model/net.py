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
from collections import namedtuple
from mxnet.gluon import HybridBlock, nn
import mxnet as mx
from gluoncv.model_zoo.mobilenetv3 import mobilenet_v3_small
from .feature import FPNFeatureExpander
from mxnet.gluon.contrib.nn import SyncBatchNorm


def resnet50_v1b(pretrained=False, root='~/.mxnet/models', ctx=mx.cpu(0), **kwargs):
    """Constructs a ResNetV1b-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    """
    from gluoncv.model_zoo.resnetv1b import ResNetV1b, BottleneckV1b

    name_prefix = 'resnetv1b_'
    if 'name_prefix' in kwargs:
        _tmp = kwargs.pop('name_prefix')
        if _tmp:
            name_prefix = _tmp + name_prefix
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], name_prefix=name_prefix, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file

        model.load_parameters(
            get_model_file('resnet%d_v%db' % (50, 1), tag=pretrained, root=root),
            ctx=ctx,
        )
        from gluoncv.data import ImageNet1kAttr

        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


Config = namedtuple('Config', ('net', 'outputs'))
base_net_configs = {
    'resnet50_v1b': Config(
        resnet50_v1b,
        outputs=[
            'layers1_relu8_fwd',
            'layers2_relu11_fwd',
            'layers3_relu17_fwd',
            'layers4_relu8_fwd',
        ],
    ),
    'mobilenetv3': Config(
        mobilenet_v3_small,
        outputs=[
            '_resunit0_seq-0-linear-batchnorm_fwd',
            '_resunit2_seq-2-linear-batchnorm_fwd',
            '_resunit7_seq-7-linear-batchnorm_fwd',
            '_resunit10_seq-10-linear-batchnorm_fwd',
        ],
    ),
}


class PSENet(HybridBlock):
    def __init__(
        self,
        base_net_name,
        num_kernels,
        scale=1,
        ctx=mx.cpu(),
        pretrained=False,
        **kwargs
    ):
        super(PSENet, self).__init__(**kwargs)
        self.num_kernels = num_kernels

        if 'prefix' in kwargs:
            kwargs['name_prefix'] = kwargs.pop('prefix')
        base_net_params = dict(
            pretrained=pretrained, norm_layer=nn.BatchNorm, ctx=ctx, **kwargs
        )
        base_net_cls = base_net_configs[base_net_name].net
        base_net_outputs = base_net_configs[base_net_name].outputs
        base_network = base_net_cls(**base_net_params)

        prefix = kwargs.get('name_prefix')
        if base_net_name == 'mobilenetv3' and prefix is None:
            prefix = base_network._prefix

        with mx.name.Prefix(prefix or ''):
            self.features = FPNFeatureExpander(
                network=base_network,
                outputs=base_net_outputs,
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
