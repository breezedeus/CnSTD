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
from unittest.mock import patch

from rapidocr import RapidOCR, EngineType, LangDet, ModelType, OCRVersion, LangRec
from rapidocr.utils.load_image import LoadImage
from rapidocr.utils.download_file import DownloadFileException
from rapidocr.ch_ppocr_det import TextDetector

from cnstd import CnStd
from cnstd.utils import set_logger
from cnstd.ppocr.rapid_detector import RapidDetector, Config

logger = set_logger()


def require_onnxruntime():
    pytest.importorskip("onnxruntime")


def skip_if_model_download_unavailable(exc):
    pytest.skip(f"rapidocr model download is unavailable in this environment: {exc}")


def test_whole_pipeline():
    require_onnxruntime()
    try:
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
    except DownloadFileException as exc:
        skip_if_model_download_unavailable(exc)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / "examples"
    img_path = example_dir / 'multi-line_cn1.png'
    result = engine(img_path)
    print(result)


def test_det():
    require_onnxruntime()
    config = Config(Config.DEFAULT_CFG)
    try:
        engine = TextDetector(config)
    except DownloadFileException as exc:
        skip_if_model_download_unavailable(exc)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / "docs"
    img_path = example_dir / "cnocr-wx.png"

    load_img = LoadImage()

    result = engine(load_img(img_path))
    print(result)


def test_rapid_detector():
    require_onnxruntime()
    # 测试直接指定模型文件路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_fp = os.path.join(root_dir, "models", "ch_PP-OCRv4_det_infer.onnx")
    try:
        detector = RapidDetector(
            model_name="ch_PP-OCRv5_det",
            # model_fp=model_fp,
        )
    except DownloadFileException as exc:
        skip_if_model_download_unavailable(exc)

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


def test_rapid_detector_sets_model_root_dir_for_new_rapidocr():
    detector_calls = []

    def fake_text_detector(config):
        detector_calls.append(config)
        return lambda img: None

    with patch("cnstd.ppocr.rapid_detector.TextDetector", side_effect=fake_text_detector):
        detector = RapidDetector(model_name="ch_PP-OCRv5_det")

    assert detector_calls
    config = detector_calls[0]
    assert config.model_root_dir == detector._model_dir
    assert config.model_path == detector._model_fp


def test_rapid_detector_respects_custom_model_root_dir():
    detector_calls = []
    custom_root_dir = "/tmp/custom-rapidocr-models"

    def fake_text_detector(config):
        detector_calls.append(config)
        return lambda img: None

    with patch("cnstd.ppocr.rapid_detector.TextDetector", side_effect=fake_text_detector):
        RapidDetector(
            model_name="ch_PP-OCRv5_det",
            model_root_dir=custom_root_dir,
        )

    assert detector_calls
    assert detector_calls[0].model_root_dir == custom_root_dir


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "SMALL"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_detector_supports_ppocrv6_config():
    detector_calls = []
    prepare_calls = []

    def fake_text_detector(config):
        detector_calls.append(config)
        return lambda img: None

    def fake_prepare_model_files(model_fp, remote_repo):
        prepare_calls.append((model_fp, remote_repo))
        return model_fp

    with patch(
        "cnstd.ppocr.rapid_detector.TextDetector",
        side_effect=fake_text_detector,
    ), patch(
        "cnstd.ppocr.rapid_detector.prepare_model_files",
        side_effect=fake_prepare_model_files,
    ):
        detector = RapidDetector(model_name="multi_PP-OCRv6_det_small")

    assert detector_calls
    config = detector_calls[0]
    assert config.ocr_version == OCRVersion.PPOCRV6
    assert config.model_type == ModelType.SMALL
    assert config.lang_type == LangDet.CH
    assert config.model_path.endswith("PP-OCRv6_det_small.onnx")
    assert config.model_root_dir == detector._model_dir
    assert prepare_calls == [
        (
            os.path.join(detector._model_dir, "PP-OCRv6_det_small.onnx"),
            "breezedeus/cnstd-ppocr-multi_PP-OCRv6_det_small",
        )
    ]


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "MEDIUM"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_detector_supports_ppocrv6_medium_config():
    detector_calls = []

    def fake_text_detector(config):
        detector_calls.append(config)
        return lambda img: None

    with patch(
        "cnstd.ppocr.rapid_detector.TextDetector",
        side_effect=fake_text_detector,
    ), patch(
        "cnstd.ppocr.rapid_detector.prepare_model_files",
        side_effect=lambda model_fp, remote_repo: model_fp,
    ):
        RapidDetector(model_name="multi_PP-OCRv6_det_medium")

    assert detector_calls[0].ocr_version == OCRVersion.PPOCRV6
    assert detector_calls[0].model_type == ModelType.MEDIUM


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "SMALL"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_detector_supports_ppocrv6_lang_override():
    detector_calls = []

    def fake_text_detector(config):
        detector_calls.append(config)
        return lambda img: None

    with patch(
        "cnstd.ppocr.rapid_detector.TextDetector",
        side_effect=fake_text_detector,
    ), patch(
        "cnstd.ppocr.rapid_detector.prepare_model_files",
        side_effect=lambda model_fp, remote_repo: model_fp,
    ):
        RapidDetector(model_name="multi_PP-OCRv6_det_small", lang_type=LangDet.EN)

    assert detector_calls[0].lang_type == LangDet.EN


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "MEDIUM"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_detector_supports_ppocrv6_string_lang_type():
    detector_calls = []

    def fake_text_detector(config):
        detector_calls.append(config)
        return lambda img: None

    with patch(
        "cnstd.ppocr.rapid_detector.TextDetector",
        side_effect=fake_text_detector,
    ), patch(
        "cnstd.ppocr.rapid_detector.prepare_model_files",
        side_effect=lambda model_fp, remote_repo: model_fp,
    ):
        RapidDetector(model_name="multi_PP-OCRv6_det_medium", lang_type="french")

    assert detector_calls[0].lang_type == "french"


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_detector_rejects_ppocrv6_multi_lang_type():
    with pytest.raises(ValueError, match="concrete lang_type"):
        RapidDetector(model_name="multi_PP-OCRv6_det_small", lang_type=LangDet.MULTI)


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "TINY"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_detector_rejects_ppocrv6_tiny_japan_lang_type():
    with pytest.raises(ValueError, match="Unsupported det.lang_type='japan'"):
        RapidDetector(model_name="multi_PP-OCRv6_det_tiny", lang_type="japan")


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_detector_rejects_unknown_ppocrv6_model():
    with pytest.raises(NotImplementedError, match="not a downloadable model"):
        RapidDetector(model_name="unknown_PP-OCRv6_det")


def test_cnstd_passes_model_root_dir_to_rapid_detector():
    detector_calls = []

    def fake_text_detector(config):
        detector_calls.append(config)
        return lambda img: None

    def fake_prepare_model_files(self, model_fp, root):
        self._model_fp = "/tmp/mock-model.onnx"
        self._model_dir = "/tmp/mock-model-dir"

    with patch("cnstd.ppocr.rapid_detector.TextDetector", side_effect=fake_text_detector):
        with patch.object(
            RapidDetector,
            "_assert_and_prepare_model_files",
            fake_prepare_model_files,
        ):
            std = CnStd(model_name="ch_PP-OCRv5_det", model_backend="onnx")

    assert isinstance(std.det_model, RapidDetector)
    assert detector_calls
    config = detector_calls[0]
    assert config.model_root_dir == std.det_model._model_dir
    assert config.model_path == std.det_model._model_fp
