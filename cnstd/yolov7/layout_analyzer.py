# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
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
# Credits to: https://github.com/WongKinYiu/yolov7, forked to https://github.com/breezedeus/yolov7

import os
import logging
from pathlib import Path
from typing import Union, Optional, Any, List, Dict, Tuple

from PIL import Image
import cv2
import numpy as np
import torch
from torch import nn
from numpy import random

from ..consts import MODEL_VERSION, ANALYSIS_SPACE, ANALYSIS_MODELS
from ..utils import data_dir, get_model_file, sort_boxes
from .yolo import Model
from .consts import CATEGORY_DICT
from .common import Conv
from .datasets import letterbox
from .general import (
    check_img_size,
    non_max_suppression,
    xyxy24p,
    scale_coords,
    box_partial_overlap,
)
from .torch_utils import select_device, time_synchronized
from .plots import plot_one_box

logger = logging.getLogger(__name__)


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


@torch.no_grad()
def attempt_load(
    categories, model_fp, cfg_fp, map_location=None,
):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    inner_model = Model(cfg_fp, ch=3, nc=len(categories), anchors=None).to(
        map_location
    )  # create
    state_dict = torch.load(model_fp, map_location=map_location)  # load
    inner_model.load_state_dict(state_dict)
    # inner_model.names = CATEGORIES

    model = Ensemble()
    model.append(inner_model.float().fuse().eval())

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % model_fp)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def dedup_boxes(one_out, threshold):
    def _to_iou_box(ori):
        return torch.tensor([ori[0][0], ori[0][1], ori[2][0], ori[2][1]]).unsqueeze(0)

    keep = [True] * len(one_out)
    for idx, info in enumerate(one_out):
        box = _to_iou_box(info['box'])
        if not keep[idx]:
            continue
        for l in range(idx + 1, len(one_out)):
            if not keep[l]:
                continue
            box2 = _to_iou_box(one_out[l]['box'])
            v1 = float(box_partial_overlap(box, box2).squeeze())
            v2 = float(box_partial_overlap(box2, box).squeeze())
            if v1 >= v2:
                if v1 >= threshold:
                    keep[l] = False
            else:
                if v2 >= threshold:
                    keep[idx] = False
                    break

    return [info for idx, info in enumerate(one_out) if keep[idx]]


class LayoutAnalyzer(object):
    def __init__(
        self,
        model_name: str = 'mfd',  # 'layout' or 'mfd'
        *,
        model_type: str = 'yolov7_tiny',  # 当前支持 [`yolov7_tiny`, `yolov7`]'
        model_backend: str = 'pytorch',
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        device: str = 'cpu',
        **kwargs,
    ):
        """

        Args:
            model_name (str): 模型类型。可选值：'mfd' 表示数学公式检测；'layout' 表示版面分析。默认值：'mfd'
            model_type (str): 模型类型。当前支持 'yolov7_tiny' 和 'yolov7'; 默认值: 'yolov7_tiny'
            model_backend (str): backend; 当前仅支持: 'pytorch'; 默认值: 'pytorch'
            model_fp (str): model file path; default: `None`, means that the default file path will be used
            root (str or Path): 模型文件所在的根目录。
                Linux/Mac下默认值为 `~/.cnstd`，表示模型文件所处文件夹类似 `~/.cnstd/1.2/analysis`
                Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnstd`。
            device (str): 'cpu', or 'gpu'; default: 'cpu'
            **kwargs ():
        """
        assert model_name in ('layout', 'mfd')
        model_backend = model_backend.lower()
        assert model_backend in ('pytorch', 'onnx')
        self._model_name = model_name
        self._model_type = model_type
        self._model_backend = model_backend

        if device.lower().strip() in ('cuda', 'cuda:0', 'gpu'):
            device = '0'
        self.device = select_device(device)

        self._assert_and_prepare_model_files(model_fp, root)
        logger.info('Use model: %s' % self._model_fp)

        self.categories = CATEGORY_DICT[self._model_name]
        self.model = attempt_load(
            self.categories,
            self._model_fp,
            cfg_fp=self._arch_yaml,
            map_location=self.device,
        )  # load FP32 model
        self.model.eval()

        self.stride = int(self.model.stride.max())  # model stride
        # self.img_size = check_img_size(image_size, s=self.stride)  # check img_size

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        VALID_MODELS = ANALYSIS_MODELS[self._model_name]
        if (self._model_type, self._model_backend) not in VALID_MODELS:
            raise NotImplementedError(
                'model %s is not supported currently'
                % ((self._model_type, self._model_backend),)
            )

        self._arch_yaml = VALID_MODELS[(self._model_type, self._model_backend)][
            'arch_yaml'
        ]
        if model_fp is not None:
            self._model_fp = model_fp
            return

        self._model_dir = os.path.join(root, MODEL_VERSION, ANALYSIS_SPACE)
        suffix = 'pt' if self._model_backend == 'pytorch' else 'onnx'
        model_fp = os.path.join(
            self._model_dir, '%s-%s.%s' % (self._model_name, self._model_type, suffix)
        )
        if not os.path.isfile(model_fp):
            logger.warning('Can NOT find model file %s' % model_fp)
            url = VALID_MODELS[(self._model_type, self._model_backend)]['url']

            get_model_file(url, self._model_dir)  # download the .zip file and unzip

        self._model_fp = model_fp

    def __call__(self, *args, **kwargs):
        """参考函数 `self.analyze()` 。"""
        return self.analyze(*args, **kwargs)

    def analyze(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        resized_shape: Union[int, Tuple[int, int]] = 700,
        box_margin: int = 2,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        对指定图片（列表）进行版面分析。

        Args:
            img_list (str or list): 待识别图片或图片列表；如果是 `np.ndarray`，则应该是shape为 `[H, W, 3]` 的 RGB 格式数组
            resized_shape (int or tuple): (H, W); 把图片resize到此大小再做分析；默认值为 `700`
            box_margin (int): 对识别出的内容框往外扩展的像素大小；默认值为 `2`
            conf_threshold (float): 分数阈值；默认值为 `0.25`
            iou_threshold (float): IOU阈值；默认值为 `0.45`
            **kwargs ():

        Returns: 一张图片的结果为一个list，其中每个元素表示识别出的版面中的一个元素，包含以下信息：
            * type: 版面元素对应的类型；可选值来自：`self.categories` ;
            * box: 版面元素对应的矩形框；np.ndarray, shape: (4, 2)，对应 box 4个点的坐标值 (x, y) ;
            * score: 得分，越高表示越可信 。

        """
        outs = []
        single = False
        if not isinstance(img_list, list):
            img_list = [img_list]
            single = True

        for img in img_list:
            img, img0 = self._preprocess_images(img, resized_shape)
            outs.append(
                self._analyze_one(img, img0, box_margin, conf_threshold, iou_threshold)
            )

        return outs[0] if single else outs

    def _preprocess_images(
        self,
        img: Union[str, Path, Image.Image, np.ndarray],
        resized_shape: Union[int, Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            img ():

        Returns: (img, img0)
            * img: RGB-formated ndarray: [3, H, W]
            * img0: BGR-formated ndarray: [H, W, 3]

        """
        if isinstance(img, (str, Path)):
            if not os.path.isfile(img):
                raise FileNotFoundError(img)
            img0 = cv2.imread(img, cv2.IMREAD_COLOR)
        elif isinstance(img, Image.Image):
            img0 = np.asarray(img.convert('RGB'), dtype='float32')
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            img0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError('type %s is not supported now' % str(type(img)))

        if isinstance(resized_shape, int):
            resized_shape = (resized_shape, resized_shape)
        img_size = [
            check_img_size(x, s=self.stride) for x in resized_shape
        ]  # check img_size
        # Padded resize
        img = letterbox(img0, img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0

    @torch.no_grad()
    def _analyze_one(
        self, img, img0, box_margin, conf_threshold, iou_threshold,
    ):
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=None,
            agnostic=False,
        )
        t3 = time_synchronized()

        one_out = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xyxy = self._expand(xyxy, box_margin, img0.shape)
                    one_out.append(
                        {
                            'type': self.categories[int(cls)],
                            'box': xyxy24p(xyxy, np.array),
                            'score': float(conf),
                        }
                    )

            logger.info(
                f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS'
            )

        one_out = sort_boxes(one_out, key='box')
        return dedup_boxes(one_out, threshold=0.1)

    def _expand(self, xyxy, box_margin, shape):
        xmin, ymin, xmax, ymax = [float(_x) for _x in xyxy]
        xmin = max(0, xmin - box_margin)
        ymin = max(0, ymin - box_margin)
        xmax = min(shape[1], xmax + box_margin)
        ymax = min(shape[0], ymax + box_margin)
        return [xmin, ymin, xmax, ymax]

    def save_img(self, img0, one_out, save_path):
        save_layout_img(img0, self.categories, one_out, save_path)


def save_layout_img(img0, categories, one_out, save_path):
    """可视化版面分析结果。"""
    if isinstance(img0, Image.Image):
        img0 = cv2.cvtColor(np.asarray(img0.convert('RGB')), cv2.COLOR_RGB2BGR)

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in categories]
    for one_box in one_out:
        _type = one_box['type']
        conf = one_box['score']
        box = one_box['box']
        xyxy = [box[0, 0], box[0, 1], box[2, 0], box[2, 1]]
        label = f'{_type} {conf:.2f}'
        plot_one_box(
            xyxy,
            img0,
            label=label,
            color=colors[categories.index(_type)],
            line_thickness=1,
        )

    cv2.imwrite(save_path, img0)
    logger.info(f" The image with the result is saved in: {save_path}")
