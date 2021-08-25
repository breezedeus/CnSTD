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

import os
import logging
from glob import glob
from pathlib import Path
from typing import Tuple, List, Dict, Union, Any, Optional

from PIL import Image
import numpy as np

from .consts import MODEL_VERSION, AVAILABLE_MODELS
from .model import gen_model
from .model.core import DetectionPredictor
from .utils import (
    data_dir,
    check_model_name,
    check_context,
    get_model_file,
    load_model_params,
    read_img,
)

logger = logging.getLogger(__name__)


class CnStd(object):
    """
    场景文字检测器（Scene Text Detection）。虽然名字中有个"Cn"（Chinese），但其实也可以轻松识别英文的。
    """

    MODEL_FILE_PREFIX = 'cnstd-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        model_name: str = 'db_resnet18',
        model_epoch: Optional[int] = None,
        *,
        auto_rotate_whole_image: bool = False,
        rotated_bbox: bool = True,
        context: str = 'cpu',
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        **kwargs,
    ):
        """
        Args:
            model_name: 模型名称。可选值为 'db_resnet18', 'db_resnet34', 'db_resnet50', 'db_mobilenet_v3'
            model_epoch: 模型迭代次数。默认为 None，表示使用系统自带的模型对应的迭代次数
            auto_rotate_whole_image: 是否自动对整张图片进行旋转调整。默认为False
            rotated_bbox: 是否支持检测带角度的文本框；默认为 True，表示支持；取值为 False 时，只检测水平或垂直的文本
            context: 'cpu', or 'gpu'。表明预测时是使用CPU还是GPU。默认为CPU
            model_fp: 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.ckpt' 文件）
            root: 模型文件所在的根目录。
                Linux/Mac下默认值为 `~/.cnstd`，表示模型文件所处文件夹类似 `~/.cnstd/1.0/db_resnet18`
                Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnstd`。
        """
        if 'name' in kwargs:
            logger.warning(
                'param `name` is useless and deprecated since version %s'
                % MODEL_VERSION
            )
        check_model_name(model_name)
        check_context(context)
        self._model_name = model_name
        self.context = context
        self.rotated_bbox = rotated_bbox

        self._model_file_prefix = '{}-{}'.format(self.MODEL_FILE_PREFIX, model_name)
        self._model_epoch = (
            model_epoch
            if model_epoch is not None
            else AVAILABLE_MODELS.get(model_name, [None])[0]
        )
        if self._model_epoch is not None:
            self._model_file_prefix = '%s-epoch=%03d' % (
                self._model_file_prefix,
                self._model_epoch,
            )

        self._assert_and_prepare_model_files(model_fp, root)

        self._model = self._get_model(auto_rotate_whole_image)
        logger.info('CnStd is initialized, with context {}'.format(self.context))

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, self._model_name)
        fps = glob('%s/%s*.ckpt' % (self._model_dir, self._model_file_prefix))
        if len(fps) > 1:
            raise ValueError(
                'multiple ckpt files are found in %s, not sure which one should be used'
                % self._model_dir
            )
        elif len(fps) < 1:
            logger.warning('no ckpt file is found in %s' % self._model_dir)
            get_model_file(self._model_dir)  # download the .zip file and unzip
            fps = glob('%s/%s*.ckpt' % (self._model_dir, self._model_file_prefix))

        self._model_fp = fps[0]

    def _get_model(self, auto_rotate_whole_image):
        logger.info('use model: %s' % self._model_fp)
        model = gen_model(
            self._model_name,
            pretrained_backbone=False,
            auto_rotate_whole_image=auto_rotate_whole_image,
            rotated_bbox=self.rotated_bbox,
        )
        model.eval()
        model.to(self.context)
        load_model_params(model, self._model_fp, self.context)

        predictor = DetectionPredictor(model, context=self.context)
        return predictor

    def detect(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        resized_shape: Tuple[int, int] = (768, 768),
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
            resized_shape: (height, width), 检测前，先把原始图片resize到此大小。默认为 `(768, 768)`。
                注：其中取值必须都能整除32。这个取值对检测结果的影响较大，可以针对自己的应用多尝试几组值，再选出最优值。
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
                   'box'：检测出的文字对应的矩形框；4个 (rotated_bbox==False) 或者 5个 (rotated_bbox==True) 元素;
                       * 4个元素时的含义：对应 rotated_bbox==False，取值为：[xmin, ymin, xmax, ymax] ;
                       * 5个元素时的含义：对应 rotated_bbox==True，取值为：[x, y, w, h, angle]。
                   'score'：得分；float 类型；分数越高表示越可靠；
                   'cropped_img'：对应'box'中的图片patch（RGB格式），会把倾斜的图片旋转为水平。
                          np.ndarray 类型，shape: (height, width, 3), 取值范围：[0, 255]；

                 示例:
                   [{'box': array([824.19433594, 712.30371094, 19.98046875, 9.99023438, -0.0]),
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

        idx = 0
        out = []
        while idx * batch_size < len(img_list):
            imgs = img_list[idx * batch_size : (idx + 1) * batch_size]
            res = self._detect_batch(
                imgs,
                resized_shape=resized_shape,
                preserve_aspect_ratio=preserve_aspect_ratio,
                min_box_size=min_box_size,
                box_score_thresh=box_score_thresh,
                **kwargs,
            )
            out.extend(res)
            idx += 1

        return out[0] if single else out

    def _detect_batch(
        self,
        img_list: List[Union[str, Path, Image.Image, np.ndarray]],
        resized_shape: Tuple[int, int],
        preserve_aspect_ratio: bool,
        min_box_size: int,
        box_score_thresh: float,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        img_list = self._preprocess_images(img_list)
        return self._model(
            img_list,
            resized_shape=resized_shape,
            preserve_aspect_ratio=preserve_aspect_ratio,
            min_box_size=min_box_size,
            box_score_thresh=box_score_thresh,
        )

    @classmethod
    def _preprocess_images(
        cls, img_list: List[Union[str, Path, Image.Image, np.ndarray]]
    ):
        out_list = []
        for img in img_list:
            if isinstance(img, (str, Path)):
                if not os.path.isfile(img):
                    raise FileNotFoundError(img)
                pil_img = read_img(img)
                out_list.append(pil_img)
            elif isinstance(img, (Image.Image, np.ndarray)):
                out_list.append(img)
            else:
                raise TypeError('type %s is not supported now' % str(type(img)))
        return out_list
