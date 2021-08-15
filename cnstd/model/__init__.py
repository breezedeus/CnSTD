# coding: utf-8
from .dbnet import *


MODEL_FUNCS = {
    'db_resnet50': db_resnet50,
    'db_resnet34': db_resnet34,
    'db_resnet18': db_resnet18,
    'db_mobilenet_v3': db_mobilenet_v3,
}


def gen_model(model_name, **kwargs):
    try:
        return MODEL_FUNCS[model_name](**kwargs)
    except KeyError as e:
        raise KeyError('got unsupported model name: %s' % model_name)
