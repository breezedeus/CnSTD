# coding: utf-8
# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.
from itertools import chain

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from unidecode import unidecode
from scipy.optimize import linear_sum_assignment
from .geometry import rbbox_to_polygon, fit_rbbox

__all__ = ['TextMatch', 'box_iou', 'box_ioa', 'mask_iou', 'rbox_to_mask',
           'nms', 'LocalizationConfusion']


def string_match(word1: str, word2: str) -> Tuple[bool, bool, bool, bool]:
    """Perform string comparison with multiple levels of tolerance

    Args:
        word1: a string
        word2: another string

    Returns:
        a tuple with booleans specifying respectively whether the raw strings, their lower-case counterparts, their
            unidecode counterparts and their lower-case unidecode counterparts match
    """
    raw_match = (word1 == word2)
    caseless_match = (word1.lower() == word2.lower())
    unidecode_match = (unidecode(word1) == unidecode(word2))

    # Warning: the order is important here otherwise the pair ("EUR", "€") cannot be matched
    unicase_match = (unidecode(word1).lower() == unidecode(word2).lower())

    return raw_match, caseless_match, unidecode_match, unicase_match


class TextMatch:
    """Implements text match metric (word-level accuracy) for recognition task.

    The raw aggregated metric is computed as follows:

    .. math::
        \\forall X, Y \\in \\mathcal{W}^N,
        TextMatch(X, Y) = \\frac{1}{N} \\sum\\limits_{i=1}^N f_{Y_i}(X_i)

    with the indicator function :math:`f_{a}` defined as:

    .. math::
        \\forall a, x \\in \\mathcal{W},
        f_a(x) = \\left\\{
            \\begin{array}{ll}
                1 & \\mbox{if } x = a \\\\
                0 & \\mbox{otherwise.}
            \\end{array}
        \\right.

    where :math:`\\mathcal{W}` is the set of all possible character sequences,
    :math:`N` is a strictly positive integer.

    Example::
        >>> metric = TextMatch()
        >>> metric.update(['Hello', 'world'], ['hello', 'world'])
        >>> metric.summary()
    """

    def __init__(self) -> None:
        self.reset()

    def update(
        self,
        gt: List[str],
        pred: List[str],
    ) -> None:
        """Update the state of the metric with new predictions

        Args:
            gt: list of groung-truth character sequences
            pred: list of predicted character sequences"""

        if len(gt) != len(pred):
            raise AssertionError("prediction size does not match with ground-truth labels size")

        for gt_word, pred_word in zip(gt, pred):
            _raw, _caseless, _unidecode, _unicase = string_match(gt_word, pred_word)
            self.raw += int(_raw)
            self.caseless += int(_caseless)
            self.unidecode += int(_unidecode)
            self.unicase += int(_unicase)

        self.total += len(gt)

    def summary(self) -> Dict[str, float]:
        """Computes the aggregated metrics

        Returns:
            a dictionary with the exact match score for the raw data, its lower-case counterpart, its unidecode
            counterpart and its lower-case unidecode counterpart
        """
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")

        return dict(
            raw=self.raw / self.total,
            caseless=self.caseless / self.total,
            unidecode=self.unidecode / self.total,
            unicase=self.unicase / self.total,
        )

    def reset(self) -> None:
        self.raw = 0
        self.caseless = 0
        self.unidecode = 0
        self.unicase = 0
        self.total = 0


def box_iou(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Compute the IoU between two sets of bounding boxes

    Args:
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)
    Returns:
        the IoU matrix of shape (N, M)
    """

    iou_mat = np.zeros((boxes_1.shape[0], boxes_2.shape[0]), dtype=np.float32)

    if boxes_1.shape[0] > 0 and boxes_2.shape[0] > 0:
        l1, t1, r1, b1 = np.split(boxes_1, 4, axis=1)
        l2, t2, r2, b2 = np.split(boxes_2, 4, axis=1)

        left = np.maximum(l1, l2.T)
        top = np.maximum(t1, t2.T)
        right = np.minimum(r1, r2.T)
        bot = np.minimum(b1, b2.T)

        intersection = np.clip(right - left, 0, np.Inf) * np.clip(bot - top, 0, np.Inf)
        union = (r1 - l1) * (b1 - t1) + ((r2 - l2) * (b2 - t2)).T - intersection
        iou_mat = intersection / union

    return iou_mat


def box_ioa(boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
    """Compute the IoA (intersection over area) between two sets of bounding boxes:
    ioa(i, j) = inter(i, j) / area(i)

    Args:
        boxes_1: bounding boxes of shape (N, 4) in format (xmin, ymin, xmax, ymax)
        boxes_2: bounding boxes of shape (M, 4) in format (xmin, ymin, xmax, ymax)
    Returns:
        the IoA matrix of shape (N, M)
    """

    ioa_mat = np.zeros((boxes_1.shape[0], boxes_2.shape[0]), dtype=np.float32)

    if boxes_1.shape[0] > 0 and boxes_2.shape[0] > 0:
        l1, t1, r1, b1 = np.split(boxes_1, 4, axis=1)
        l2, t2, r2, b2 = np.split(boxes_2, 4, axis=1)

        left = np.maximum(l1, l2.T)
        top = np.maximum(t1, t2.T)
        right = np.minimum(r1, r2.T)
        bot = np.minimum(b1, b2.T)

        intersection = np.clip(right - left, 0, np.Inf) * np.clip(bot - top, 0, np.Inf)
        area = (r1 - l1) * (b1 - t1)
        ioa_mat = intersection / area

    return ioa_mat


def mask_iou(masks_1: np.ndarray, masks_2: np.ndarray) -> np.ndarray:
    """Compute the IoU between two sets of boolean masks

    Args:
        masks_1: boolean masks of shape (N, H, W)
        masks_2: boolean masks of shape (M, H, W)

    Returns:
        the IoU matrix of shape (N, M)
    """

    if masks_1.shape != masks_2.shape:
        raise AssertionError("both boolean masks should have the same spatial shape")

    iou_mat = np.zeros((masks_1.shape[0], ), dtype=np.float32)

    if masks_1.shape[0] > 0 and masks_2.shape[0] > 0:
        intersection = np.logical_and(masks_1, masks_2)
        union = np.logical_or(masks_1, masks_2)
        iou_mat = intersection.sum(axis=(1, 2)) / (union.sum(axis=(1, 2)) + 1e-6)

    return iou_mat


def rbox_to_mask(boxes_list: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Convert boxes to masks

    Args:
        boxes_list: list of rotated bounding boxes of shape (M, 5) in format (x, y, w, h, alpha)
        shape: spatial shapes of the output masks

    Returns:
        the boolean masks of shape (N, H, W)
    """

    batch_size = len(boxes_list)
    masks = np.zeros((batch_size, *shape), dtype=np.uint8)

    for idx, boxes in enumerate(boxes_list):
        if boxes.shape[0] > 0:
            # Get absolute coordinates
            if boxes.dtype != np.int:
                abs_boxes = boxes.copy()
                abs_boxes = abs_boxes.round().astype(np.int)
            else:
                abs_boxes = boxes
                abs_boxes[:, 2:] = abs_boxes[:, 2:] + 1

            # TODO: optimize slicing to improve vectorization
            for _box in abs_boxes:
                box = rbbox_to_polygon(_box)
                cv2.fillPoly(masks[idx], [np.array(box, np.int32)], 1)

    return masks.astype(bool)


def nms(boxes: np.ndarray, thresh: float = .5) -> List[int]:
    """Perform non-max suppression, borrowed from <https://github.com/rbgirshick/fast-rcnn>`_.

    Args:
        boxes: np array of straight boxes: (*, 5), (xmin, ymin, xmax, ymax, score)
        thresh: iou threshold to perform box suppression.

    Returns:
        A list of box indexes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class LocalizationConfusion:
    """Implements common confusion metrics and mean IoU for localization evaluation.

    The aggregated metrics are computed as follows:

    .. math::
        \\forall Y \\in \\mathcal{B}^N, \\forall X \\in \\mathcal{B}^M, \\\\
        Recall(X, Y) = \\frac{1}{N} \\sum\\limits_{i=1}^N g_{X}(Y_i) \\\\
        Precision(X, Y) = \\frac{1}{M} \\sum\\limits_{i=1}^N g_{X}(Y_i) \\\\
        meanIoU(X, Y) = \\frac{1}{M} \\sum\\limits_{i=1}^M \\max\\limits_{j \\in [1, N]}  IoU(X_i, Y_j)

    with the function :math:`IoU(x, y)` being the Intersection over Union between bounding boxes :math:`x` and
    :math:`y`, and the function :math:`g_{X}` defined as:

    .. math::
        \\forall y \\in \\mathcal{B},
        g_X(y) = \\left\\{
            \\begin{array}{ll}
                1 & \\mbox{if } y\\mbox{ has been assigned to any }(X_i)_i\\mbox{ with an }IoU \\geq 0.5 \\\\
                0 & \\mbox{otherwise.}
            \\end{array}
        \\right.

    where :math:`\\mathcal{B}` is the set of possible bounding boxes,
    :math:`N` (number of ground truths) and :math:`M` (number of predictions) are strictly positive integers.

    Example::

    Args:
        iou_thresh: minimum IoU to consider a pair of prediction and ground truth as a match
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        rotated_bbox: bool = False,
        mask_shape: Tuple[int, int] = (1024, 1024),
    ) -> None:
        self.iou_thresh = iou_thresh
        self.rotated_bbox = rotated_bbox
        self.mask_shape = mask_shape
        self.reset()

    def update(self, gt_boxes: List[List[np.ndarray]], norm_preds: List[np.ndarray]) -> Tuple[float, float]:
        """

        Args:
            gt_boxes: 这里面的值是未归一化到 [0, 1] 的
            norm_preds: 这里面的值是归一化到 [0, 1] 的

        Returns:

        """
        gts = self._transform_gt_polygons(gt_boxes)
        preds = []
        for n_pred in norm_preds:
            pred = n_pred.copy()
            pred[:, [0, 2]] *= self.mask_shape[1]
            pred[:, [1, 3]] *= self.mask_shape[0]
            preds.append(pred)

        cur_iou, cur_matches = 0.0, 0.0
        batch_size = len(preds)
        if batch_size > 0:
            # Compute IoU
            if self.rotated_bbox:
                mask_gts = rbox_to_mask(gts, shape=self.mask_shape)
                mask_preds = rbox_to_mask(preds, shape=self.mask_shape)
                iou_vec = mask_iou(mask_gts, mask_preds)
                cur_iou = iou_vec.sum()
                cur_matches = int((iou_vec >= self.iou_thresh).sum())
            else:
                iou_mat = box_iou(gts, preds)
                cur_iou = float(iou_mat.max(axis=1).sum())

                # Assign pairs
                gt_indices, pred_indices = linear_sum_assignment(-iou_mat)
                cur_matches = int((iou_mat[gt_indices, pred_indices] >= self.iou_thresh).sum())

        self.tot_iou += cur_iou
        self.matches += cur_matches
        # Update counts
        self.num_gts += batch_size

        cur_match = cur_matches / (1e-6 + batch_size)
        cur_iou = cur_iou / (1e-6 + batch_size)
        return cur_match, cur_iou

    def _transform_gt_polygons(self, polgons: List[List[np.ndarray]]) -> List[np.ndarray]:
        """

        Args:
            polgons: 最里层每个 np.ndarray 是个 [4, 2] 的矩阵，表示一个box的4个点的坐标。

        Returns:
            list of rotated bounding boxes of shape (M, 5) in format (x, y, w, h, alpha)

        """
        out = []
        for boxes in polgons:
            new_boxes = []
            for box in boxes:
                box = box.astype(np.uint)
                new_boxes.append(fit_rbbox(box) if self.rotated_bbox else cv2.boundingRect(box))
            out.append(np.asarray(new_boxes))
        return out

    def summary(self) -> Tuple[Optional[float], Optional[float]]:
        """Computes the aggregated metrics

        Returns:
            a tuple with the recall, precision and meanIoU scores
        """

        # match
        match = self.matches / (1e-6 + self.num_gts)

        # mean IoU
        mean_iou = self.tot_iou / (1e-6 + self.num_gts)

        return match, mean_iou

    def reset(self) -> None:
        self.num_gts = 0
        self.matches = 0
        self.tot_iou = 0.
