# coding: utf-8
import os
import sys
import pytest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnstd.model.dbnet import db_resnet18, db_resnet34, db_resnet50, db_mobilenet_v3

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(root_dir, 'examples')


def test_db_resnet():
    model = db_mobilenet_v3(pretrained=False)
    input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    out = model(input_tensor)
    print(out.keys())
    print(out['preds'][0][0].shape)
    breakpoint()
