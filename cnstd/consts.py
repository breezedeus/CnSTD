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

from pathlib import Path
from typing import Dict, Any
from copy import deepcopy
from collections import OrderedDict

from torchvision.models import (
    resnet50,
    resnet34,
    resnet18,
    mobilenet_v3_large,
    mobilenet_v3_small,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)

from .__version__ import __version__


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '1.0.*'，对应的 MODEL_VERSION 都是 '1.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
VOCAB_FP = Path(__file__).parent.parent / 'label_cn.txt'

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'db_resnet50': {
        'backbone': resnet50,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_channels': [256, 512, 1024, 2048],
        'input_shape': (3, 768, 768),  # resize后输入模型的图片大小, 即 `resized_shape`
        'url': None,
    },
    'db_resnet34': {
        'backbone': resnet34,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_channels': [64, 128, 256, 512],
        'input_shape': (3, 768, 768),
        'url': None,
    },
    'db_resnet18': {
        'backbone': resnet18,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_channels': [64, 128, 256, 512],
        'input_shape': (3, 768, 768),
        'url': None,
    },
    'db_mobilenet_v3': {
        'backbone': mobilenet_v3_large,
        'backbone_submodule': 'features',
        'fpn_layers': ['3', '6', '12', '16'],
        'fpn_channels': [24, 40, 112, 960],
        'input_shape': (3, 768, 768),
        'url': None,
    },
    'db_mobilenet_v3_small': {
        'backbone': mobilenet_v3_small,
        'backbone_submodule': 'features',
        'fpn_layers': ['1', '3', '8', '12'],
        'fpn_channels': [16, 24, 48, 576],
        'input_shape': (3, 768, 768),
        'url': None,
    },
    'db_shufflenet_v2': {
        'backbone': shufflenet_v2_x2_0,
        'backbone_submodule': None,
        'fpn_layers': ['maxpool', 'stage2', 'stage3', 'stage4'],
        'fpn_channels': [24, 244, 488, 976],
        'input_shape': (3, 768, 768),
        'url': None,
    },
    'db_shufflenet_v2_small': {
        'backbone': shufflenet_v2_x1_5,
        'backbone_submodule': None,
        'fpn_layers': ['maxpool', 'stage2', 'stage3', 'stage4'],
        'fpn_channels': [24, 176, 352, 704],
        'input_shape': (3, 768, 768),
        'url': None,
    },
    'db_shufflenet_v2_tiny': {
        'backbone': shufflenet_v2_x1_0,
        'backbone_submodule': None,
        'fpn_layers': ['maxpool', 'stage2', 'stage3', 'stage4'],
        'fpn_channels': [24, 116, 232, 464],
        'input_shape': (3, 768, 768),
        'url': None,
    },
}

root_url = (
    'https://beiye-model.oss-cn-beijing.aliyuncs.com/models/cnstd/%s/' % MODEL_VERSION
)
# name: (epochs, url)
# 免费模型
FREE_MODELS = OrderedDict(
    {
        'db_resnet34': {
            'model_epoch': 41,
            'fpn_type': 'pan',
            'url': root_url + 'db_resnet34-pan.zip',
        },
        'db_resnet18': {
            'model_epoch': 34,
            'fpn_type': 'pan',
            'url': root_url + 'db_resnet18-pan.zip',
        },
        'db_mobilenet_v3': {
            'model_epoch': 47,
            'fpn_type': 'pan',
            'url': root_url + 'db_mobilenet_v3-pan.zip',
        },
        'db_mobilenet_v3_small': {
            'model_epoch': 37,
            'fpn_type': 'pan',
            'url': root_url + 'db_mobilenet_v3_small-pan.zip',
        },
        'db_shufflenet_v2': {
            'model_epoch': 41,
            'fpn_type': 'pan',
            'url': root_url + 'db_shufflenet_v2-pan.zip',
        },
        'db_shufflenet_v2_small': {
            'model_epoch': 34,
            'fpn_type': 'pan',
            'url': root_url + 'db_shufflenet_v2_small-pan.zip',
        },
    }
)

# 付费模型
PAID_MODELS = OrderedDict(
    {
        'db_shufflenet_v2_tiny': {
            'model_epoch': 48,
            'fpn_type': 'pan',
            'url': root_url + 'db_shufflenet_v2_tiny-pan.zip',
        },
    }
)

AVAILABLE_MODELS = deepcopy(FREE_MODELS)
AVAILABLE_MODELS.update(PAID_MODELS)
