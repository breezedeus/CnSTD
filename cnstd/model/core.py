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
# Credits: adapted from https://github.com/mindee/doctr

import logging
from typing import List, Any, Optional, Dict, Tuple, Union, Callable

import numpy as np
import cv2
from PIL import Image
import torch

from ..transforms import Resize
from ..utils import (
    pil_to_numpy,
    normalize_img_array,
    imsave,
    get_resized_ratio,
)
from ..utils.repr import NestedObject
from ..utils._utils import rotate_page, get_bitmap_angle, extract_crops, extract_rcrops


logger = logging.getLogger(__name__)

__all__ = ['DetectionModel', 'DetectionPostProcessor', 'DetectionPredictor']


class DetectionModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        auto_rotate_whole_image: whether to detect the angle of the whold image and calibrate it automatically
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time
        rotated_bbox: whether to detect non-vertical and non-horizontal boxes
    """

    def __init__(
        self,
        *,
        auto_rotate_whole_image=False,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.5,
        rotated_bbox: bool = False,
    ) -> None:

        self.auto_rotate_whole_image = auto_rotate_whole_image
        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.rotated_bbox = rotated_bbox

    def extra_repr(self) -> str:
        return f"box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(
        pred: np.ndarray, points: np.ndarray, rotated_bbox: bool = False
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]

        if not rotated_bbox:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin : ymax + 1, xmin : xmax + 1].mean()

        else:
            mask = np.zeros((h, w), np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def bitmap_to_boxes(self, pred: np.ndarray, bitmap: np.ndarray,) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, proba_map: np.ndarray,) -> Tuple[List[np.ndarray], List[float]]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, H, W)

        returns:
            list of N tensors (for each input sample), with each tensor of shape (*, 5) or (*, 6),
            and a list of N angles (page orientations).
        """

        bitmap = (proba_map > self.bin_thresh).astype(proba_map.dtype)

        boxes_batch, angles_batch = [], []
        # Kernel for opening, empirical law for ksize
        k_size = 1 + int(proba_map[0].shape[0] / 512)
        kernel = np.ones((k_size, k_size), np.uint8)

        for p_, bitmap_ in zip(proba_map, bitmap):
            # Perform opening (erosion + dilatation)
            bitmap_ = cv2.morphologyEx(bitmap_, cv2.MORPH_OPEN, kernel)
            # Rotate bitmap and proba_map
            if self.auto_rotate_whole_image:
                angle = get_bitmap_angle(bitmap_)
            else:
                angle = 0.0
            angles_batch.append(angle)
            bitmap_, p_ = rotate_page(bitmap_, -angle), rotate_page(p_, -angle)
            boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
            boxes_batch.append(boxes)

        return boxes_batch, angles_batch


class DetectionPredictor(NestedObject):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    _children_names: List[str] = ['model']

    def __init__(self, model, *, context='cpu') -> None:
        self.device = torch.device(context)
        self.model = model
        self.model.eval()
        self.extract_crops_fn = (
            extract_rcrops if self.model.rotated_bbox else extract_crops
        )

    @torch.no_grad()
    def __call__(
        self,
        img_list: List[Union[Image.Image, np.ndarray]],
        resized_shape: Tuple[int, int],
        preserve_aspect_ratio: bool = True,
        min_box_size: int = 8,
        box_score_thresh: float = 0.5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """

        Args:
            img_list: list, which element's should be one type of Image.Image and np.ndarray.
                For Image.Image, it should be generated from read_img;
                For np.ndarray, it should be RGB-style, with shape [H, W, 3], scale [0, 255]
            resized_shape: tuple, [height, width], height and width after resizing original images
            preserve_aspect_ratio: whether or not presserve aspect ratio of original images when resizing them
            min_box_size: minimal size of detected boxes; boxes with smaller height or width will be ignored
            box_score_thresh: score threshold for boxes, boxes with scores lower than this value will be ignored
            **kwargs:

        Returns:

        """
        if len(img_list) == 0:
            return []
        size_transform = Resize(
            resized_shape, preserve_aspect_ratio=preserve_aspect_ratio
        )
        ori_imgs, batch, compress_ratios = self.preprocess(
            img_list, resized_shape, size_transform, preserve_aspect_ratio
        )

        out = self.model(batch, return_preds=True, **kwargs)
        boxes, angles = out['preds']
        results = []
        for image, _boxes, compress_ratio, angle in zip(
            ori_imgs, boxes, compress_ratios, angles
        ):
            # image = restore_img(image.numpy().transpose((1, 2, 0)))  # res: [H, W, 3]
            image = image.transpose((1, 2, 0)).astype(np.uint8)  # res: [H, W, 3]
            rotated_img = np.ascontiguousarray(rotate_page(image, -angle))

            _scores = _boxes[:, -1].tolist()
            _boxes = _boxes[:, :-1]
            # resize back
            _boxes[:, [0, 2]] /= compress_ratio[1]
            _boxes[:, [1, 3]] /= compress_ratio[0]

            out_boxes = _boxes.copy()
            out_boxes[:, [0, 2]] *= rotated_img.shape[1]
            out_boxes[:, [1, 3]] *= rotated_img.shape[0]

            one_out = []
            for crop, score, box in zip(
                self.extract_crops_fn(rotated_img, _boxes), _scores, out_boxes
            ):
                if score < box_score_thresh:
                    continue
                if min(crop.shape[:2]) < min_box_size:
                    continue
                one_out.append(dict(box=box, score=score, cropped_img=crop))
            results.append({'rotated_angle': angle, 'detected_texts': one_out})

        return results

    def preprocess(
        self,
        pil_img_list: List[Union[Image.Image, np.ndarray]],
        resized_shape: Tuple[int, int],
        size_transform: Callable,
        preserve_aspect_ratio: bool,
    ) -> Tuple[List[np.ndarray], torch.Tensor, List[Tuple[float, float]]]:
        ori_img_list = []
        img_list = []
        compress_ratios_list = []
        for img in pil_img_list:
            if isinstance(img, Image.Image):
                img = pil_to_numpy(img)  # res: np.ndarray, RGB-style, [3, H, W]
            elif isinstance(img, np.ndarray) and img.shape[2] == 3:
                img = img.transpose((2, 0, 1))  # [H, W, 3] to [3, H, W]
            else:
                raise ValueError('unsupported image input is found')

            ori_img_list.append(img)
            compress_ratio = self._compress_ratio(
                img.shape[1:], resized_shape, preserve_aspect_ratio
            )
            compress_ratios_list.append(compress_ratio)
            img = size_transform(torch.from_numpy(img)).numpy()
            img = normalize_img_array(img)
            img_list.append(torch.from_numpy(img))
        return (
            ori_img_list,
            torch.stack(img_list, dim=0).to(device=self.device),
            compress_ratios_list,
        )

    def _compress_ratio(self, ori_hw, target_hw, preserve_aspect_ratio):
        if not preserve_aspect_ratio:
            return 1.0, 1.0

        resized_ratios = get_resized_ratio(ori_hw, target_hw, True)
        ratio = resized_ratios[0]
        ori_h, ori_w = ori_hw
        target_h, target_w = target_hw
        return ratio * ori_h / target_h, ratio * ori_w / target_w
