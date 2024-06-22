# coding: utf-8
# Copyright (C) 2022-2024, [Breezedeus](https://www.breezedeus.com).
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
# YOLO Detector based on Ultralytics.

from pathlib import Path
from typing import Union, Optional, Any, List, Dict, Tuple
import logging

from PIL import Image
import numpy as np
from ultralytics import YOLO

from .utils import sort_boxes, dedup_boxes, xyxy24p, select_device, expand_box_by_margin

logger = logging.getLogger(__name__)


class YoloDetector(object):
    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        static_resized_shape: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        YOLO Detector based on Ultralytics.
        Args:
            model_path (optional str): model path, default is None.
            device (optional str): device to use, default is None.
            static_resized_shape (optional int or tuple): static resized shape, default is None.
                When it is not None, the input image will be resized to this shape before detection,
                ignoring the input parameter `resized_shape` if .detect() is called.
                Some format of models may require a fixed input size, such as CoreML.
            **kwargs (): other parameters.
        """
        self.device = select_device(device)
        self.static_resized_shape = static_resized_shape
        self.model = YOLO(model_path, task='detect')

    def __call__(self, *args, **kwargs):
        """参考函数 `self.detect()` 。"""
        return self.detect(*args, **kwargs)

    def detect(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        resized_shape: int = 768,
        box_margin: int = 0,
        conf: float = 0.25,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        对指定图片（列表）进行目标检测。

        Args:
            img_list (str or list): 待识别图片或图片列表；如果是 `np.ndarray`，则应该是shape为 `[H, W, 3]` 的 RGB 格式数组
            resized_shape (int or tuple): (H, W); 把图片resize到此大小再做分析；默认值为 `700`
            box_margin (int): 对识别出的内容框往外扩展的像素大小；默认值为 `2`
            conf (float): 分数阈值；默认值为 `0.25`
            **kwargs (): 其他预测使用的参数，以及以下值
                - dedup_thrsh: 去重时使用的阈值；默认值为 `0.1`

        Returns: 一张图片的结果为一个list，其中每个元素表示识别出的版面中的一个元素，包含以下信息：
            * type: 版面元素对应的类型；可选值来自：`self.categories` ;
            * box: 版面元素对应的矩形框；np.ndarray, shape: (4, 2)，对应 box 4个点的坐标值 (x, y) ;
            * score: 得分，越高表示越可信 。

        """
        dedup_thrsh = kwargs.pop('dedup_thrsh') if 'dedup_thrsh' in kwargs else 0.1
        single = not isinstance(img_list, (list, tuple))
        # Ultralytics 需要的 ndarray 是 HWC，BGR 格式
        if isinstance(img_list, np.ndarray):
            img_list = img_list[:, :, ::-1]
        elif isinstance(img_list, list):
            img_list = [
                img[:, :, ::-1] if isinstance(img, np.ndarray) else img
                for img in img_list
            ]

        if self.static_resized_shape is not None:
            resized_shape = self.static_resized_shape
        batch_results = self.model.predict(
            img_list, imgsz=resized_shape, conf=conf, device=self.device, **kwargs
        )
        outs = []
        for res in batch_results:
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            scores = res.boxes.conf.cpu().numpy().tolist()
            labels = res.boxes.cls.cpu().int().numpy().tolist()
            categories = res.names
            height, width = res.orig_shape
            one_out = []
            for box, score, label in zip(boxes, scores, labels):
                box = expand_box_by_margin(box, box_margin, (height, width))
                box = xyxy24p(box, ret_type=np.array)
                one_out.append({'box': box, 'score': score, 'type': categories[label]})

            one_out = sort_boxes(one_out, key='box')
            one_out = dedup_boxes(one_out, threshold=dedup_thrsh)
            outs.append(one_out)

        if single and len(outs) == 1:
            return outs[0]
        return outs
