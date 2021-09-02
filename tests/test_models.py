# coding: utf-8
import os
import sys
import pytest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(root_dir, 'examples')

from cnstd.consts import MODEL_CONFIGS
from cnstd.model.dbnet import gen_dbnet


def test_db_mobilenet():
    model = gen_dbnet(MODEL_CONFIGS['db_mobilenet_v3'], pretrained=False, pretrained_backbone=False)
    input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    out = model(input_tensor)
    print(out.keys())
    print(out['preds'][0][0].shape)


def test_db_shufflenet():
    model = gen_dbnet(MODEL_CONFIGS['db_shufflenet_v2'], pretrained=False, pretrained_backbone=False)
    input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    out = model(input_tensor)
    print(out.keys())
    print(out['preds'][0][0].shape)
