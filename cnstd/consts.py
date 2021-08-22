# coding: utf-8
# Copyright (C) 2021, Breezedeus.
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

from torchvision.models import resnet50, resnet34, resnet18, mobilenet_v3_large

from .__version__ import __version__


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '1.0.*'，对应的 MODEL_VERSION 都是 '1.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
VOCAB_FP = Path(__file__).parent.parent / 'label_cn.txt'
# BACKBONE_NET_NAME = ['mobilenetv3', 'resnet50_v1b']
# BACKBONE_CONFIGS = {
#
# }

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
}

root_url = (
        'https://static.einplus.cn/cnstd/%s'
        % MODEL_VERSION
)
# name: (epochs, url)
AVAILABLE_MODELS = {
    'db_resnet18': (59, root_url + '/mobilenetv3.zip'),
    'db_resnet34': (49, root_url + '/resnet50_v1b.zip'),
    'db_mobilenet_v3': (49, root_url + '/resnet50_v1b.zip'),
}
