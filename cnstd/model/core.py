# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
from typing import List, Any, Optional, Dict, Tuple, Union

import numpy as np
import cv2
from PIL import Image
import torch

from ..transforms import Resize
from ..utils import (
    pil_to_numpy,
    normalize_img_array,
    restore_img,
    imsave,
    get_resized_ratio,
)
from ..utils.repr import NestedObject
from .._utils import rotate_page, get_bitmap_angle, extract_crops, extract_rcrops


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
        min_size_box (int): minimal length (pix) to keep a box
        max_candidates (int): maximum boxes to consider in a single page
        box_thresh (float): minimal objectness score to consider a box
    """

    def __init__(
        self,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.5,
        rotated_bbox: bool = False,
    ) -> None:

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
            angle = get_bitmap_angle(bitmap_)
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

    def __init__(
        self,
        model,
        *,
        resized_shape,
        min_box_size=8,
        preserve_aspect_ratio=True,
        debug=False,
    ) -> None:

        self.resized_shape = resized_shape
        self.min_box_size = min_box_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.debug = debug
        self.model = model
        self.model.eval()
        self.val_transform = Resize(
            self.resized_shape, preserve_aspect_ratio=self.preserve_aspect_ratio
        )
        self.extract_crops_fn = (
            extract_rcrops if self.model.rotated_bbox else extract_crops
        )

    @torch.no_grad()
    def __call__(
        self,
        img_list: List[Union[Image.Image, np.ndarray]],
        box_score_thresh: float = 0.5,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """

        Args:
            img_list: list, which element's should be one type of Image.Image and np.ndarray.
                For Image.Image, it should be generated from read_img;
                For np.ndarray, it should be RGB-style, with shape [3, H, W], scale [0, 255]
            box_score_thresh: score threshold for boxes, boxes with scores lower than this value will be ignored
            **kwargs:

        Returns:

        """
        ori_imgs, batch, compress_ratios = self.preprocess(img_list)

        out = self.model(batch, return_preds=True, **kwargs)  # type:ignore[operator]
        boxes, angles = out['preds']
        crops_list = []
        scores_list = []
        boxes_list = []
        idx = 0
        for image, _boxes, compress_ratio, angle in zip(
            ori_imgs, boxes, compress_ratios, angles
        ):
            # image = restore_img(image.numpy().transpose((1, 2, 0)))  # res: [H, W, 3]
            image = image.transpose((1, 2, 0)).astype(np.uint8)  # res: [H, W, 3]
            rotated_img = np.ascontiguousarray(rotate_page(image, -angle))
            crops = []
            scores = []
            clean_boxes = []

            _scores = _boxes[:, -1].tolist()
            _boxes = _boxes[:, :-1]
            # resize back
            _boxes[:, [0, 2]] /= compress_ratio[1]
            _boxes[:, [1, 3]] /= compress_ratio[0]

            for crop, score, box in zip(
                self.extract_crops_fn(rotated_img, _boxes), _scores, _boxes
            ):
                if score < box_score_thresh:
                    continue
                if min(crop.shape[:2]) < self.min_box_size:
                    continue
                crops.append(crop)
                scores.append(score)
                clean_boxes.append(box)
            crops_list.append(crops)
            scores_list.append(scores)
            boxes_list.append(clean_boxes)

            if self.debug:
                self._plot_for_debugging(
                    rotated_img, crops, _boxes, _scores, box_score_thresh, idx
                )
            idx += 1

        return crops_list, scores_list, boxes_list

    def _plot_for_debugging(
        self, rotated_img, crops, _boxes, _scores, box_score_thresh, idx
    ):
        import matplotlib.pyplot as plt
        import math

        logger.info('%d boxes are found' % len(crops))
        ncols = 3
        nrows = math.ceil(len(crops) / ncols)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        for i, axi in enumerate(ax.flat):
            if i >= len(crops):
                break
            axi.imshow(crops[i])
        plt.tight_layout(True)
        plt.savefig('crops-%d.png' % idx)

        if _boxes.dtype != np.int:
            _boxes[:, [0, 2]] *= rotated_img.shape[1]
            _boxes[:, [1, 3]] *= rotated_img.shape[0]

        for box, score in zip(_boxes, _scores):
            if score < box_score_thresh:  # score < 0.5
                continue
            if len(box) == 5:  # rotated_box == True
                x, y, w, h, alpha = box.astype('float32')
                box = cv2.boxPoints(((x, y), (w, h), alpha))
                box = np.int0(box)
                cv2.drawContours(rotated_img, [box], 0, (255, 0, 0), 2)
            else:  # len(box) == 4, rotated_box == False
                xmin, ymin, xmax, ymax = box.astype('float32')
                cv2.rectangle(rotated_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        imsave(rotated_img, 'result-%d.png' % idx, normalized=False)

    def preprocess(
        self, pil_img_list: List[Union[Image.Image, np.ndarray]]
    ) -> Tuple[List[np.ndarray], torch.Tensor, List[Tuple[float, float]]]:
        ori_img_list = []
        img_list = []
        compress_ratios_list = []
        for img in pil_img_list:
            if isinstance(img, Image.Image):
                img = pil_to_numpy(img)  # res: np.ndarray, RGB-style, [3, H, W]
                ori_img_list.append(img)
            compress_ratio = self._compress_ratio(img.shape[1:], self.resized_shape)
            compress_ratios_list.append(compress_ratio)
            img = self.val_transform(torch.from_numpy(img)).numpy()
            img = normalize_img_array(img)
            img_list.append(torch.from_numpy(img))
        return ori_img_list, torch.stack(img_list, dim=0), compress_ratios_list

    def _compress_ratio(self, ori_hw, target_hw):
        if not self.preserve_aspect_ratio:
            return 1.0, 1.0

        resized_ratios = get_resized_ratio(ori_hw, target_hw, True)
        ratio = resized_ratios[0]
        ori_h, ori_w = ori_hw
        target_h, target_w = target_hw
        return ratio * ori_h / target_h, ratio * ori_w / target_w
