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

import os
import pytest
import torch
from pathlib import Path

from rapidocr import RapidOCR, EngineType, LangDet, ModelType, OCRVersion, LangRec
from rapidocr.utils import LoadImage
from rapidocr.ch_ppocr_det import TextDetector

from cnstd.utils import set_logger
from cnstd.ppocr.rapid_detector import RapidDetector, Config

logger = set_logger()


def test_whole_pipeline():
    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.SERVER,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": ModelType.SERVER,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
    )
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / "examples"
    img_path = example_dir / 'multi-line_cn1.png'
    result = engine(img_path)
    print(result)


def test_det():
    config = Config(Config.DEFAULT_CFG)
    engine = TextDetector(config)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / "docs"
    img_path = example_dir / "cnocr-wx.png"

    load_img = LoadImage()

    result = engine(load_img(img_path))
    print(result)


def test_rapid_detector():
    # 测试直接指定模型文件路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_fp = os.path.join(root_dir, "models", "ch_PP-OCRv4_det_infer.onnx")
    detector = RapidDetector(
        model_name="ch_PP-OCRv5_det",
        # model_fp=model_fp,
    )

    example_dir = Path(root_dir) / "docs"
    img_path = example_dir / "cnocr-wx.png"

    result = detector.detect(img_path)
    print(result)
    assert isinstance(result, dict)
    assert "rotated_angle" in result
    assert "detected_texts" in result
    assert isinstance(result["detected_texts"], list)
    if len(result["detected_texts"]) > 0:
        box = result["detected_texts"][0]
        assert "box" in box
        assert "score" in box
        assert box["box"].shape == (4, 2)
        assert isinstance(box["score"], float)

    # 测试使用默认参数
    detector = RapidDetector()
    result = detector.detect(img_path)
    print(result)
    assert isinstance(result, dict)
    assert "rotated_angle" in result
    assert "detected_texts" in result

    # 测试错误的模型名称
    with pytest.raises(NotImplementedError):
        RapidDetector(model_name="invalid")
