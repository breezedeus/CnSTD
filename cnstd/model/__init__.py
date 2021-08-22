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

from copy import deepcopy

from .dbnet import gen_dbnet, DBNet
from ..consts import MODEL_CONFIGS


def gen_model(model_name: str, pretrained_backbone: bool = True, **kwargs) -> DBNet:
    """

    Args:
        model_name:
        pretrained_backbone: whether use pretrained for the backbone model
        **kwargs:
            'rotated_bbox': bool, 是否考虑非水平的boxes
            'pretrained': bool, 是否使用预训练好的模型
            'input_shape': Tuple[int, int, int], resize后输入模型的图片大小：[C, H, W]

    Returns: a DBNet model

    """
    if model_name not in MODEL_CONFIGS:
        raise KeyError('got unsupported model name: %s' % model_name)

    config = deepcopy(MODEL_CONFIGS[model_name])
    config.update(**kwargs)
    return gen_dbnet(
        config, pretrained_backbone=pretrained_backbone, **kwargs
    )
