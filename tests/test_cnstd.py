# coding: utf-8
import os
import sys
import pytest
import mxnet as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnstd import CnStd
from cnstd.consts import AVAILABLE_MODELS

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(root_dir, 'examples')


def test_mobilenetv3():
    model_name = 'mobilenetv3'
    img_fp = os.path.join(example_dir, 'beauty2.jpg')
    std = CnStd(model_name)
    box_info_list = std.detect(img_fp)

    img = mx.image.imread(img_fp, 1)
    box_info_list2 = std.detect(img)
    assert len(box_info_list) == len(box_info_list2)


def test_resnet50_v1b():
    model_name = 'resnet50_v1b'
    img_fp = os.path.join(example_dir, 'beauty2.jpg')
    std = CnStd(model_name)
    box_info_list = std.detect(img_fp)

    img = mx.image.imread(img_fp, 1)
    box_info_list2 = std.detect(img)
    assert len(box_info_list) == len(box_info_list2)


INSTANCE_ID = 0


@pytest.mark.parametrize('model_name', AVAILABLE_MODELS.keys())
def test_multiple_instances(model_name):
    global INSTANCE_ID
    print('test multiple instances for model_name: %s' % model_name)
    img_fp = os.path.join(example_dir, 'beauty2.jpg')
    INSTANCE_ID += 1
    print('instance id: %d' % INSTANCE_ID)
    std1 = CnStd(model_name, name='instance-%d' % INSTANCE_ID)
    box_info_list = std1.detect(img_fp)
    INSTANCE_ID += 1
    std2 = CnStd(model_name, name='instance-%d' % INSTANCE_ID)
    box_info_list2 = std2.detect(img_fp)
    assert len(box_info_list) == len(box_info_list2)
