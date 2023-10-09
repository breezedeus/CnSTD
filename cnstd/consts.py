# coding: utf-8
# Copyright (C) 2021-2023, [Breezedeus](https://github.com/breezedeus).
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
import logging
from pathlib import Path
from typing import Tuple, Set, Dict, Any, Optional, Union
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

logger = logging.getLogger(__name__)


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '1.0.*'，对应的 MODEL_VERSION 都是 '1.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
VOCAB_FP = Path(__file__).parent.parent / 'label_cn.txt'
# Which OSS source will be used for downloading model files, 'CN' or 'HF'
DOWNLOAD_SOURCE = os.environ.get('CNSTD_DOWNLOAD_SOURCE', 'CN')

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

HF_HUB_REPO_ID = "breezedeus/cnstd-cnocr-models"
HF_HUB_SUBFOLDER = "models/cnstd/%s" % MODEL_VERSION
CN_OSS_ENDPOINT = (
    "https://sg-models.oss-cn-beijing.aliyuncs.com/cnstd/%s/" % MODEL_VERSION
)


def format_hf_hub_url(url: str) -> dict:
    return {
        'repo_id': HF_HUB_REPO_ID,
        'subfolder': HF_HUB_SUBFOLDER,
        'filename': url,
        'cn_oss': CN_OSS_ENDPOINT,
    }


class AvailableModels(object):
    CNSTD_SPACE = '__cnstd__'

    # name: (epochs, url)
    # 免费模型
    FREE_MODELS = OrderedDict(
        {
            ('db_resnet34', 'pytorch'): {
                'model_epoch': 41,
                'fpn_type': 'pan',
                'url': 'db_resnet34-pan.zip',
            },
            ('db_resnet18', 'pytorch'): {
                'model_epoch': 34,
                'fpn_type': 'pan',
                'url': 'db_resnet18-pan.zip',
            },
            ('db_mobilenet_v3', 'pytorch'): {
                'model_epoch': 47,
                'fpn_type': 'pan',
                'url': 'db_mobilenet_v3-pan.zip',
            },
            ('db_mobilenet_v3_small', 'pytorch'): {
                'model_epoch': 37,
                'fpn_type': 'pan',
                'url': 'db_mobilenet_v3_small-pan.zip',
            },
            ('db_shufflenet_v2', 'pytorch'): {
                'model_epoch': 41,
                'fpn_type': 'pan',
                'url': 'db_shufflenet_v2-pan.zip',
            },
            ('db_shufflenet_v2_small', 'pytorch'): {
                'model_epoch': 34,
                'fpn_type': 'pan',
                'url': 'db_shufflenet_v2_small-pan.zip',
            },
        }
    )

    # 付费模型
    PAID_MODELS = OrderedDict(
        {
            ('db_shufflenet_v2_tiny', 'pytorch'): {
                'model_epoch': 48,
                'fpn_type': 'pan',
                'url': 'db_shufflenet_v2_tiny-pan.zip',
            },
        }
    )

    CNSTD_MODELS = deepcopy(FREE_MODELS)
    CNSTD_MODELS.update(PAID_MODELS)

    OUTER_MODELS = {}

    def all_models(self) -> Set[Tuple[str, str]]:
        return set(self.CNSTD_MODELS.keys()) | set(self.OUTER_MODELS.keys())

    def __contains__(self, model_name_backend: Tuple[str, str]) -> bool:
        return model_name_backend in self.all_models()

    def register_models(self, model_dict: Dict[Tuple[str, str], Any], space: str):
        assert not space.startswith('__')
        for key, val in model_dict.items():
            if key in self.CNSTD_MODELS or key in self.OUTER_MODELS:
                logger.warning(
                    'model %s has already existed, and will be ignored' % key
                )
                continue
            val = deepcopy(val)
            val['space'] = space
            self.OUTER_MODELS[key] = val

    def get_space(self, model_name, model_backend) -> Optional[str]:
        if (model_name, model_backend) in self.CNSTD_MODELS:
            return self.CNSTD_SPACE
        elif (model_name, model_backend) in self.OUTER_MODELS:
            return self.OUTER_MODELS[(model_name, model_backend)]['space']
        return None

    def get_value(self, model_name, model_backend, key) -> Optional[Any]:
        if (model_name, model_backend) in self.CNSTD_MODELS:
            info = self.CNSTD_MODELS[(model_name, model_backend)]
        elif (model_name, model_backend) in self.OUTER_MODELS:
            info = self.OUTER_MODELS[(model_name, model_backend)]
        else:
            logger.warning(
                'no url is found for model %s' % ((model_name, model_backend),)
            )
            return None
        return info.get(key)

    def get_epoch(self, model_name, model_backend) -> Optional[int]:
        return self.get_value(model_name, model_backend, 'model_epoch')

    def get_fpn_type(self, model_name, model_backend) -> Optional[int]:
        return self.get_value(model_name, model_backend, 'fpn_type')

    def get_url(self, model_name, model_backend) -> Optional[dict]:
        url = self.get_value(model_name, model_backend, 'url')
        if url:
            url = format_hf_hub_url(url)

        return url


AVAILABLE_MODELS = AvailableModels()

ANGLE_CLF_SPACE = 'angle_clf'
ANGLE_CLF_MODELS = {
    ('ch_ppocr_mobile_v2.0_cls', 'onnx'): {
        'url': format_hf_hub_url('ch_ppocr_mobile_v2.0_cls_infer-onnx.zip')
    }
}

ANALYSIS_SPACE = 'analysis'
ANALYSIS_MODELS = {
    'layout': {
        ('yolov7_tiny', 'pytorch'): {
            'url': format_hf_hub_url('yolov7_tiny_layout-pytorch.zip'),
            'arch_yaml': Path(__file__).parent / 'yolov7' / 'yolov7-tiny-layout.yaml',
        }
    },
    'mfd': {
        ('yolov7_tiny', 'pytorch'): {
            'url': format_hf_hub_url('yolov7_tiny_mfd-pytorch.zip'),
            'arch_yaml': Path(__file__).parent / 'yolov7' / 'yolov7-tiny-mfd.yaml',
        },
        ('yolov7', 'pytorch'): {
            'url': format_hf_hub_url('yolov7_mfd-pytorch.zip'),
            'arch_yaml': Path(__file__).parent / 'yolov7' / 'yolov7-mfd.yaml',
        },
    },
}
