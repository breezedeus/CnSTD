# coding: utf-8
# Copyright (C) 2024, [Breezedeus](https://github.com/breezedeus).
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
from copy import deepcopy
from typing import Tuple, List, Dict, Union, Any, Optional

import numpy as np
from PIL import Image
import cv2
# from rapidocr_onnxruntime.ch_ppocr_det import TextDetector
# from rapidocr_onnxruntime import RapidOCR
from rapidocr import EngineType, LangDet, ModelType, OCRVersion
from rapidocr.utils.typings import TaskType
from rapidocr.ch_ppocr_det import TextDetector

from ..consts import AVAILABLE_MODELS, MODEL_VERSION
from ..utils import read_img, data_dir, prepare_model_files
from .utility import get_rotate_crop_image
from .consts import PP_SPACE

logger = logging.getLogger(__name__)

class Config(dict):
    DEFAULT_CFG = {
        "engine_type": EngineType.ONNXRUNTIME,
        "lang_type": LangDet.CH,
        "model_type": ModelType.SERVER,
        "ocr_version": OCRVersion.PPOCRV5,
        "task_type": TaskType.DET,
        "model_path": None,
        "model_dir": None,
        "limit_side_len": 736,
        "limit_type": "min",
        "std": [0.5, 0.5, 0.5],
        "mean": [0.5, 0.5, 0.5],
        "thresh": 0.3,
        "box_thresh": 0.5,
        "max_candidates": 1000,
        "unclip_ratio": 1.6,
        "use_dilation": True,
        "score_mode": "fast",
        "engine_cfg": {
            "intra_op_num_threads": -1,
            "inter_op_num_threads": -1,
            "enable_cpu_mem_arena": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "use_cuda": False,
            "cuda_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
            "use_dml": False,
            "dm_ep_cfg": None,
            "use_cann": False,
            "cann_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "npu_mem_limit": 21474836480,
                "op_select_impl_mode": "high_performance",
                "optypelist_for_implmode": "Gelu",
                "enable_cann_graph": True,
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            if isinstance(v, dict):
                v = Config(v)
            self[k] = v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class RapidDetector(object):
    """
    场景文字检测器（Scene Text Detection），使用 rapidocr_onnxruntime 中的 TextDetector。
    """

    def __init__(
        self,
        model_name: str = 'ch_PP-OCRv4_det',
        *,
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        context: str = 'cpu',  # ['cpu', 'gpu']
        limit_side_len: int = 736,
        limit_type: str = "min",
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
        use_dilation: bool = True,
        score_mode: str = "fast",
        **kwargs,
    ):
        """
        Args:
            model_name: 模型名称，目前只支持 'rapid'，默认为 'rapid'
            model_fp: 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.onnx' 文件）
            root: 模型文件所在的根目录。默认为 `~/.cnstd`
            context: 使用的设备，可选值为 'cpu' 或 'gpu'，默认为 'cpu'
            limit_side_len: 限制图片最长边的长度，默认为 736
            limit_type: 限制类型，可选值为 'min' 或 'max'，默认为 'min'
            thresh: 二值化阈值，默认为 0.3
            box_thresh: 文本框阈值，默认为 0.5
            max_candidates: 最大候选框数量，默认为 1000
            unclip_ratio: 文本框扩张比例，默认为 1.6
            use_dilation: 是否使用膨胀，默认为 True
            score_mode: 得分模式，可选值为 'fast' 或 'slow'，默认为 'fast'
            kwargs: 其他参数
        """
        self._model_name = model_name
        self._model_backend = 'onnx'
        self._assert_and_prepare_model_files(model_fp, root)
        use_gpu = context.lower() not in ('cpu', 'mps')

        config = Config.DEFAULT_CFG
        config["engine_cfg"]["use_cuda"] = use_gpu
        if "engine_cfg" in kwargs:
            config["engine_cfg"].update(kwargs["engine_cfg"])

        config.update({
            "limit_side_len": limit_side_len,
            "limit_type": limit_type,
            "thresh": thresh,
            "box_thresh": box_thresh,
            "max_candidates": max_candidates,
            "unclip_ratio": unclip_ratio,
            "use_dilation": use_dilation,
            "score_mode": score_mode,
            "model_path": self._model_fp,
        })
        # 从 model_name 中获取 model_type 和 ocr_version
        config["model_type"] = ModelType.SERVER if "server" in model_name else ModelType.MOBILE
        config["ocr_version"] = OCRVersion.PPOCRV5 if "v5" in model_name else OCRVersion.PPOCRV4

        config = Config(config)
        self._detector = TextDetector(config)

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, PP_SPACE, self._model_name)
        model_fp = os.path.join(self._model_dir, '%s_infer.onnx' % self._model_name)
        if not os.path.isfile(model_fp):
            logger.warning('can not find model file %s' % model_fp)
            if (self._model_name, self._model_backend) not in AVAILABLE_MODELS:
                raise NotImplementedError(
                    '%s is not a downloadable model'
                    % ((self._model_name, self._model_backend),)
                )
            remote_repo = AVAILABLE_MODELS.get_value(self._model_name, self._model_backend, 'repo')
            model_fp = prepare_model_files(model_fp, remote_repo)

        self._model_fp = model_fp
        logger.info('use model: %s' % self._model_fp)

    def detect(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        检测图片中的文本。
        Args:
            img_list: 支持对单个图片或者多个图片（列表）的检测。每个值可以是图片路径，或者已经读取进来 PIL.Image.Image 或 np.ndarray,
                格式应该是 RGB 3通道，shape: (height, width, 3), 取值：[0, 255]
            kwargs: 其他参数，目前未被使用

        Returns:
            List[Dict], 每个Dict对应一张图片的检测结果。Dict 中包含以下 keys：
               * 'rotated_angle': float, 整张图片旋转的角度。只有 auto_rotate_whole_image==True 才可能非0。
               * 'detected_texts': list, 每个元素存储了检测出的一个框的信息，使用词典记录，包括以下几个值：
                   'box'：检测出的文字对应的矩形框；np.ndarray, shape: (4, 2)，对应 box 4个点的坐标值 (x, y) ;
                   'score'：得分；float 类型；分数越高表示越可靠；
                   'cropped_img'：对应'box'中的图片patch（RGB格式），会把倾斜的图片旋转为水平。
                          np.ndarray 类型，shape: (height, width, 3), 取值范围：[0, 255]；
        """
        single = False
        if isinstance(img_list, (list, tuple)):
            pass
        elif isinstance(img_list, (str, Path, Image.Image, np.ndarray)):
            img_list = [img_list]
            single = True
        else:
            raise TypeError('type %s is not supported now' % str(type(img_list)))

        out = []
        for img in img_list:
            if isinstance(img, (str, Path)):
                if not os.path.isfile(img):
                    raise FileNotFoundError(img)
                img = read_img(img)
            if isinstance(img, Image.Image):
                img = np.array(img)

            if not isinstance(img, np.ndarray):
                raise TypeError('type %s is not supported now' % str(type(img)))

            # rapidocr 需要 BGR 格式的图片
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[..., ::-1]  # RGB to BGR

            det_out = self._detector(img)
            if det_out is None or len(det_out.boxes) < 1:
                out.append({
                    'rotated_angle': 0.0,  # rapidocr 不支持自动旋转
                    'detected_texts': [],
                })
                continue

            # boxes = self._detector.sorted_boxes(boxes)

            # 构造返回结果
            detected_texts = []
            for box, score in zip(det_out.boxes, det_out.scores):
                box = np.array(box).astype(np.int32)
                img_crop = get_rotate_crop_image(img, deepcopy(box))
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                detected_texts.append({
                    'box': box,
                    'score': score,
                    'cropped_img': img_crop.astype('uint8'),
                })

            out.append({
                'rotated_angle': 0.0,  # rapidocr 不支持自动旋转
                'detected_texts': detected_texts,
            })

        return out[0] if single else out
