# coding: utf-8
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnstd import CnStd
from cnstd.consts import AVAILABLE_MODELS

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(root_dir, 'examples')


def test_ppocr_models():
    model_name, model_backend = 'ch_PP-OCRv3_det', 'onnx'
    img_fp = os.path.join(example_dir, 'beauty2.jpg')
    std = CnStd(model_name, model_backend=model_backend, use_angle_clf=True)
    box_info_list = std.detect(img_fp)
    print(len(box_info_list))


@pytest.mark.parametrize('model_name, model_backend', AVAILABLE_MODELS.all_models())
def test_cnstd(model_name, model_backend):
    img_fp = os.path.join(example_dir, 'beauty2.jpg')
    std = CnStd(model_name, model_backend=model_backend)
    box_info_list = std.detect(img_fp)
    print(len(box_info_list))
