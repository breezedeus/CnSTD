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

import numpy as np
import cv2
from math import floor
from typing import List
from statistics import median_low

__all__ = ['estimate_orientation', 'extract_crops', 'extract_rcrops', 'rotate_page', 'get_bitmap_angle']


def extract_crops(img: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
    """Created cropped images from list of bounding boxes

    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)

    Returns:
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 4:
        raise AssertionError("boxes are expected to be relative and in order (xmin, ymin, xmax, ymax)")

    # Project relative coordinates
    _boxes = boxes.copy()
    if _boxes.dtype != np.int:
        _boxes[:, [0, 2]] *= img.shape[1]
        _boxes[:, [1, 3]] *= img.shape[0]
        _boxes = _boxes.round().astype(int)
        # Add last index
        _boxes[2:] += 1
        _boxes[:2] -= 1
        _boxes[_boxes < 0] = 0
    return [img[box[1]: box[3], box[0]: box[2]] for box in _boxes]


def extract_rcrops(img: np.ndarray, boxes: np.ndarray, dtype=np.float32) -> List[np.ndarray]:
    """Created cropped images from list of rotated bounding boxes

    Args:
        img: input image
        boxes: bounding boxes of shape (N, 5) where N is the number of boxes, and the relative
            coordinates (x, y, w, h, alpha)

    Returns:
        list of cropped images
    """
    if boxes.shape[0] == 0:
        return []
    if boxes.shape[1] != 5:
        raise AssertionError("boxes are expected to be relative and in order (x, y, w, h, alpha)")

    # Project relative coordinates
    _boxes = boxes.copy()
    if _boxes.dtype != np.int:
        _boxes[:, [0, 2]] *= img.shape[1]
        _boxes[:, [1, 3]] *= img.shape[0]

    crops = []
    for box in _boxes:
        x, y, w, h, alpha = box.astype(dtype)
        vertical_box = False
        if (abs(alpha) < 3 and w * 1.3 < h) or (90 - abs(alpha) < 3 and w > h * 1.3):
            vertical_box = True

        process_func = _process_vertical_box if vertical_box else _process_horizontal_box
        crop = process_func(img, box, dtype)

        crops.append(crop)

    return crops


def _process_horizontal_box(img, box, dtype):
    x, y, w, h, alpha = box.astype(dtype)
    clockwise = False
    if w > h:
        clockwise = True
    if clockwise:
        #  1 -------- 2
        #  |          |
        #  * -------- 3
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=dtype)
    else:
        #  * -------- 1
        #  |          |
        #  3 -------- 2
        # dst_pts = np.array([[h - 1, 0], [h - 1, w - 1], [0, w - 1]], dtype=dtype)
        #  2 -------- 3
        #  |          |
        #  1 -------- *
        dst_pts = np.array([[0, w - 1], [0, 0], [h - 1, 0]], dtype=dtype)
    # The transformation matrix
    src_pts = cv2.boxPoints(((x, y), (w, h), alpha))
    M = cv2.getAffineTransform(src_pts[1:, :], dst_pts)
    # Warp the rotated rectangle
    if clockwise:
        crop = cv2.warpAffine(img, M, (int(w), int(h)))
    else:
        crop = cv2.warpAffine(img, M, (int(h), int(w)))
    return crop


def _process_vertical_box(img, box, dtype):
    x, y, w, h, alpha = box.astype(dtype)
    clockwise = False
    if w > h:
        clockwise = True
    if clockwise:
        #  2 ------- 3
        #  |         |
        #  |         |
        #  |         |
        #  |         |
        #  1 ------- *
        dst_pts = np.array([[0, w - 1], [0, 0], [h - 1, 0]], dtype=dtype)
        # dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=dtype)
    else:
        #  1 ------- 2
        #  |         |
        #  |         |
        #  |         |
        #  |         |
        #  * ------- 3
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=dtype)
    # The transformation matrix
    src_pts = cv2.boxPoints(((x, y), (w, h), alpha))
    M = cv2.getAffineTransform(src_pts[1:, :], dst_pts)
    # Warp the rotated rectangle
    if clockwise:
        crop = cv2.warpAffine(img, M, (int(h), int(w)))
    else:
        crop = cv2.warpAffine(img, M, (int(w), int(h)))
    return crop


def rotate_page(
        image: np.ndarray,
        angle: float = 0.,
        min_angle: float = 1.
) -> np.ndarray:
    """Rotate an image counterclockwise by an ange alpha (negative angle to go clockwise).

    Args:
        image: numpy tensor to rotate
        angle: rotation angle in degrees, between -90 and +90
        min_angle: min. angle in degrees to rotate a page

    Returns:
        Rotated array or tf.Tensor, padded by 0 by default.
    """
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return image

    height, width = image.shape[:2]
    center = (height / 2, width / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (width, height))


def get_max_width_length_ratio(contour: np.ndarray) -> float:
    """
    Get the maximum shape ratio of a contour.
    Args:
        contour: the contour from cv2.findContour

    Returns: the maximum shape ratio

    """
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w / h, h / w)


def estimate_orientation(img: np.ndarray, n_ct: int = 50, ratio_threshold_for_lines: float = 5) -> float:
    """Estimate the angle of the general document orientation based on the
     lines of the document and the assumption that they should be horizontal.

        Args:
            img: the img to analyze
            n_ct: the number of contours used for the orientation estimation
            ratio_threshold_for_lines: this is the ratio w/h used to discriminates lines
        Returns:
            the angle of the general document orientation
        """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    thresh = cv2.threshold(gray_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # try to merge words in lines
    (h, w) = img.shape[:2]
    k_x = max(1, (floor(w / 100)))
    k_y = max(1, (floor(h / 100)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, k_y))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours
    contours = sorted(contours, key=get_max_width_length_ratio, reverse=True)

    angles = []
    for contour in contours[:n_ct]:
        _, (w, h), angle = cv2.minAreaRect(contour)
        if w / h > ratio_threshold_for_lines:  # select only contours with ratio like lines
            angles.append(angle)
        elif w / h < 1 / ratio_threshold_for_lines:  # if lines are vertical, substract 90 degree
            angles.append(angle - 90)
    return -median_low(angles)


def get_bitmap_angle(bitmap: np.ndarray, n_ct: int = 20, std_max: float = 3.) -> float:
    """From a binarized segmentation map, find contours and fit min area rectangles to determine page angle

    Args:
        bitmap: binarized segmentation map
        n_ct: number of contours to use to fit page angle
        std_max: maximum deviation of the angle distribution to consider the mean angle reliable

    Returns:
        The angle of the page
    """
    # Find all contours on binarized seg map
    contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find largest contours and fit angles
    # Track heights and widths to find aspect ratio (determine is rotation is clockwise)
    angles, heights, widths = [], [], []
    for ct in contours[:n_ct]:
        _, (w, h), alpha = cv2.minAreaRect(ct)
        widths.append(w)
        heights.append(h)
        angles.append(alpha)

    if np.std(angles) > std_max:
        # Edge case with angles of both 0 and 90°, or multi_oriented docs
        angle = 0.
    else:
        angle = -np.mean(angles)
        # Determine rotation direction (clockwise/counterclockwise)
        # Angle coverage: [-90°, +90°], half of the quadrant
        if np.sum(widths) < np.sum(heights):  # CounterClockwise
            angle = 90 + angle

    return angle
