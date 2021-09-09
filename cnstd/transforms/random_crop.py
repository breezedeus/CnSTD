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
# Credits: adapted from https://github.com/MhLiao/DB

from copy import deepcopy
from typing import List, Dict, Any

from PIL import Image
import numpy as np


def random_crop(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    max_tries,
    w_axis,
    h_axis,
    min_crop_side_ratio,
):
    """随机选取一个框，然后只保留这个框中图片"""
    w, h = image.size
    selected_boxes = []
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy)
        ymax = np.max(yy)
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if (
            xmax - xmin < min_crop_side_ratio * w
            or ymax - ymin < min_crop_side_ratio * h
        ):
            # area too small
            continue
        if len(boxes) != 0:
            selected_boxes = np.array(
                [
                    idx
                    for idx, box in enumerate(boxes)
                    if _in_area(box['poly'], xmin, xmax, ymin, ymax)
                ]
            )
            if len([i for i in selected_boxes if boxes[i]['text'] != '###']) > 0:
                break
        else:
            selected_boxes = []
            break
    if i == max_tries - 1:
        return image, boxes

    new_image = image.crop((xmin, ymin, xmax, ymax))
    new_boxes = []
    for i in selected_boxes:
        box = deepcopy(boxes[i])
        box['poly'][:, 0] -= xmin
        box['poly'][:, 1] -= ymin
        new_boxes.append(box)
    return new_image, new_boxes


def _in_area(box, xmin, xmax, ymin, ymax) -> bool:
    box_axis_in_area = (
        (box[:, 0] >= xmin)
        & (box[:, 0] <= xmax)
        & (box[:, 1] >= ymin)
        & (box[:, 1] <= ymax)
    )
    return np.sum(box_axis_in_area) == 4
