# coding: utf-8
from typing import Dict, Any

from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v3_large

from .dbnet import gen_dbnet, DBNet


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'db_resnet50': {
        'backbone': resnet50,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_channels': [256, 512, 1024, 2048],
        'input_shape': (3, 1024, 1024),  # 期望的输入图片大小
        'mean': (0.5, 0.5, 0.5),
        'std': (1.0, 1.0, 1.0),
        'url': None,
    },
    'db_resnet34': {
        'backbone': resnet34,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_channels': [64, 128, 256, 512],
        'input_shape': (3, 1024, 1024),
        'mean': (0.5, 0.5, 0.5),
        'std': (1.0, 1.0, 1.0),
        'url': None,
    },
    'db_resnet18': {
        'backbone': resnet18,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_channels': [64, 128, 256, 512],
        'input_shape': (3, 1024, 1024),
        'mean': (0.5, 0.5, 0.5),
        'std': (1.0, 1.0, 1.0),
        'url': None,
    },
    'db_resnet18_small': {
        'backbone': resnet18,
        'backbone_submodule': None,
        'fpn_layers': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_channels': [64, 128, 256, 512],
        'input_shape': (3, 512, 512),
        'mean': (0.5, 0.5, 0.5),
        'std': (1.0, 1.0, 1.0),
        'url': None,
    },
    'db_mobilenet_v3': {
        'backbone': mobilenet_v3_large,
        'backbone_submodule': 'features',
        'fpn_layers': ['3', '6', '12', '16'],
        'fpn_channels': [24, 40, 112, 960],
        'input_shape': (3, 1024, 1024),
        'mean': (0.5, 0.5, 0.5),
        'std': (1.0, 1.0, 1.0),
        'url': None,
    },
}


def gen_model(model_name: str, pretrained_backbone: bool = True, **kwargs) -> DBNet:
    """

    Args:
        model_name:
        pretrained_backbone: whether use pretrained for the backbone model
        **kwargs:
            'rotated_bbox': bool, 是否考虑非水平的boxes
            'pretrained': bool, 是否使用预训练好的模型

    Returns: a DBNet model

    """
    if model_name not in MODEL_CONFIGS:
        raise KeyError('got unsupported model name: %s' % model_name)

    return gen_dbnet(
        MODEL_CONFIGS[model_name], pretrained_backbone=pretrained_backbone, **kwargs
    )
