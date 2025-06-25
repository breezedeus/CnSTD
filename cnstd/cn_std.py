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

from __future__ import absolute_import

import logging
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Union, Any, Optional

from PIL import Image
import numpy as np

from .consts import AVAILABLE_MODELS
from .detector import Detector
from .ppocr import PP_SPACE, PPDetector, RapidDetector
from .ppocr.angle_classifier import AngleClassifier
from .utils import data_dir

logger = logging.getLogger(__name__)


class CnStd(object):
    """
    场景文字检测器（Scene Text Detection）。虽然名字中有个"Cn"（Chinese），但其实也可以轻松识别英文的。
    """

    def __init__(
        self,
        model_name: str = 'ch_PP-OCRv5_det',
        *,
        auto_rotate_whole_image: bool = False,
        rotated_bbox: bool = True,
        context: str = 'cpu',
        model_fp: Optional[str] = None,
        model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        root: Union[str, Path] = data_dir(),
        use_angle_clf: bool = False,
        angle_clf_configs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Args:
            model_name: 模型名称。默认为 'ch_PP-OCRv5_det'
            auto_rotate_whole_image: 是否自动对整张图片进行旋转调整。默认为False
            rotated_bbox: 是否支持检测带角度的文本框；默认为 True，表示支持；取值为 False 时，只检测水平或垂直的文本
            context: 'cpu', or 'gpu'。表明预测时是使用CPU还是GPU。默认为CPU
            model_fp: 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.ckpt' 文件）
            model_backend (str): 'pytorch', or 'onnx'。表明预测时是使用 PyTorch 版本模型，还是使用 ONNX 版本模型。
                同样的模型，ONNX 版本的预测速度一般是 PyTorch 版本的2倍左右。默认为 'onnx'。
            root: 模型文件所在的根目录。
                Linux/Mac下默认值为 `~/.cnstd`，表示模型文件所处文件夹类似 `~/.cnstd/1.2/db_resnet18`
                Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnstd`。
            use_angle_clf (bool): 对于检测出的文本框，是否使用角度分类模型进行调整（检测出的文本框可能会存在倒转180度的情况）。
                默认为 `False`
            angle_clf_configs (dict): 角度分类模型对应的参数取值，主要包含以下值：
                - model_name: 模型名称。默认为 'ch_ppocr_mobile_v2.0_cls'
                - model_fp: 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.onnx' 文件）。默认为 `None`
                具体可参考类 `AngleClassifier` 的说明
        """
        self.space = AVAILABLE_MODELS.get_space(model_name, model_backend)
        if self.space is None:
            logger.warning(
                'no available model is found for name %s and backend %s'
                % (model_name, model_backend)
            )
            model_backend = 'onnx' if model_backend == 'pytorch' else 'pytorch'
            logger.warning(
                'trying to use name %s and backend %s' % (model_name, model_backend)
            )
            self.space = AVAILABLE_MODELS.get_space(model_name, model_backend)

        if self.space == AVAILABLE_MODELS.CNSTD_SPACE:
            det_cls = Detector
        elif self.space == PP_SPACE:
            det_name = AVAILABLE_MODELS.get_value(model_name, model_backend, 'detector')
            det_cls = RapidDetector if det_name == 'RapidDetector' else PPDetector
        else:
            raise NotImplementedError(
                '%s is not supported currently' % ((model_name, model_backend),)
            )

        self.det_model = det_cls(
            model_name=model_name,
            auto_rotate_whole_image=auto_rotate_whole_image,
            rotated_bbox=rotated_bbox,
            context=context,
            model_fp=model_fp,
            model_backend=model_backend,
            root=root,
            **kwargs,
        )

        self.use_angle_clf = use_angle_clf
        if self.use_angle_clf:
            angle_clf_configs = angle_clf_configs or dict()
            angle_clf_configs['root'] = root
            self.angle_clf = AngleClassifier(**angle_clf_configs)

    def detect(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        resized_shape: Union[int, Tuple[int, int]] = (768, 768),
        preserve_aspect_ratio: bool = True,
        min_box_size: int = 8,
        box_score_thresh: float = 0.3,
        batch_size: int = 20,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        检测图片中的文本。
        Args:
            img_list: 支持对单个图片或者多个图片（列表）的检测。每个值可以是图片路径，或者已经读取进来 PIL.Image.Image 或 np.ndarray,
                格式应该是 RGB 3通道，shape: (height, width, 3), 取值：[0, 255]
            resized_shape: `int` or `tuple`, `tuple` 含义为 (height, width), `int` 则表示高宽都为此值；
                检测前，先把原始图片resize到接近此大小（只是接近，未必相等）。默认为 `(768, 768)`。
                注：这个取值对检测结果的影响较大，可以针对自己的应用多尝试几组值，再选出最优值。
                    例如 (512, 768), (768, 768), (768, 1024)等。
            preserve_aspect_ratio: 对原始图片resize时是否保持高宽比不变。默认为 `True`。
            min_box_size: 如果检测出的文本框高度或者宽度低于此值，此文本框会被过滤掉。默认为 `8`，也即高或者宽低于 `8` 的文本框会被过滤去掉。
            box_score_thresh: 过滤掉得分低于此值的文本框。默认为 `0.3`。
            batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `20`。
            kwargs: 保留参数，目前未被使用。

        Returns:
            List[Dict], 每个Dict对应一张图片的检测结果。Dict 中包含以下 keys：
               * 'rotated_angle': float, 整张图片旋转的角度。只有 auto_rotate_whole_image==True 才可能非0。
               * 'detected_texts': list, 每个元素存储了检测出的一个框的信息，使用词典记录，包括以下几个值：
                   'box'：检测出的文字对应的矩形框；np.ndarray, shape: (4, 2)，对应 box 4个点的坐标值 (x, y) ;
                   'score'：得分；float 类型；分数越高表示越可靠；
                   'cropped_img'：对应'box'中的图片patch（RGB格式），会把倾斜的图片旋转为水平。
                          np.ndarray 类型，shape: (height, width, 3), 取值范围：[0, 255]；

                 示例:
                   [{'box': array([[416,  77],
                                   [486,  13],
                                   [800, 325],
                                   [730, 390]], dtype=int32),
                   'score': 0.8, 'cropped_img': array([[[25, 20, 24],
                                                          [26, 21, 25],
                                                          [25, 20, 24],
                                                          ...,
                                                          [11, 11, 13],
                                                          [11, 11, 13],
                                                          [11, 11, 13]]], dtype=uint8)},
                    ...
              ]

        """
        single = False
        if isinstance(img_list, (list, tuple)):
            pass
        elif isinstance(img_list, (str, Path, Image.Image, np.ndarray)):
            img_list = [img_list]
            single = True
        else:
            raise TypeError('type %s is not supported now' % str(type(img_list)))

        outs = self.det_model.detect(
            img_list,
            resized_shape=calibrate_resized_shape(resized_shape),
            preserve_aspect_ratio=preserve_aspect_ratio,
            min_box_size=min_box_size,
            box_score_thresh=box_score_thresh,
            batch_size=batch_size,
        )

        if self.use_angle_clf:
            for out in outs:
                crop_img_list = [info['cropped_img'] for info in out['detected_texts']]
                try:
                    crop_img_list, angle_list = self.angle_clf(crop_img_list)
                    for info, crop_img in zip(out['detected_texts'], crop_img_list):
                        info['cropped_img'] = crop_img
                except Exception as e:
                    logger.info(traceback.format_exc())
                    logger.info(e)

        return outs[0] if single else outs


def calibrate_resized_shape(resized_shape):
    if isinstance(resized_shape, int):
        resized_shape = (resized_shape, resized_shape)

    def calibrate(ori):
        return max(int(round(ori / 32) * 32), 32)

    return calibrate(resized_shape[0]), calibrate(resized_shape[1])
