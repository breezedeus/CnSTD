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
from collections import OrderedDict
from typing import Dict, Any

import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

from ..utils import imsave

logger = logging.getLogger(__name__)


class MakeICDARData(object):
    def __init__(self, debug=False, **kwargs):
        self.debug = debug

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        polygons = []
        ignore_tags = []
        annotations = data['polys']
        for annotation in annotations:
            polygons.append(np.array(annotation['points']))
            # polygons.append(annotation['points'])
            ignore_tags.append(annotation['ignore'])
        ignore_tags = np.array(ignore_tags, dtype=np.uint8)
        filename = ''  # data.get('filename', data['data_id'])
        if self.debug:
            boxed_image = self.draw_polygons(
                data['image'].copy(), polygons, ignore_tags
            )
            imsave(boxed_image, 'debug-polygons.jpg', normalized=False)
        # shape = np.array(data['shape'])
        return OrderedDict(
            image=data['image'],
            polygons=polygons,
            ignore_tags=ignore_tags,
            # shape=shape,
            filename=filename,
            is_training=data['is_training'],
        )

    def draw_polygons(self, image, polygons, ignore_tags):
        for i in range(len(polygons)):
            polygon = polygons[i].reshape(-1, 2).astype(np.int32)
            ignore = ignore_tags[i]
            if ignore:
                color = (255, 0, 0)  # depict ignorable polygons in red
            else:
                color = (0, 0, 255)  # depict polygons in blue

            image = np.ascontiguousarray(image)
            cv2.polylines(image, [polygon], True, color, 1)
        return image

    polylines = staticmethod(draw_polygons)


class MakeSegDetectionData(object):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = 8
    shrink_ratio = 0.4

    def __init__(self, **kwargs):
        pass
        # self.load_all(**kwargs)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        """
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']

        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(polygons, ignore_tags, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            # height = min(np.linalg.norm(polygon[0] - polygon[3]),
            #              np.linalg.norm(polygon[1] - polygon[2]))
            # width = min(np.linalg.norm(polygon[0] - polygon[1]),
            #             np.linalg.norm(polygon[2] - polygon[3]))
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = (
                    polygon_shape.area
                    * (1 - np.power(self.shrink_ratio, 2))
                    / polygon_shape.length
                )
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

        if filename is None:
            filename = ''
        data.update(image=image, polygons=polygons, gt=gt, mask=mask, filename=filename)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (
                polygon[next_index, 1] + polygon[i, 1]
            )

        return edge / 2.0


class MakeBorderMap(object):
    r"""
    Making the border map from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """
    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7

    def __init__(self, **kwargs):
        pass

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        required keys:
            image, polygons, ignore_tags
        adding keys:
            thresh_map, thresh_mask
        """
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(polygons[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = (
            polygon_shape.area
            * (1 - np.power(self.shrink_ratio, 2))
            / polygon_shape.length
        )
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width)
        )
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width)
        )

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1] = np.fmax(
            1
            - distance_map[
                ymin_valid - ymin : ymax_valid - ymax + height,
                xmin_valid - xmin : xmax_valid - xmax + width,
            ],
            canvas[ymin_valid : ymax_valid + 1, xmin_valid : xmax_valid + 1],
        )

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(
            point_1[1] - point_2[1]
        )

        cosin = (square_distance - square_distance_1 - square_distance_2) / (
            2 * np.sqrt(square_distance_1 * square_distance_2) + 1e-6
        )
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(
            square_distance_1
            * square_distance_2
            * square_sin
            / (square_distance + 1e-6)
        )

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[
            cosin < 0
        ]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (
            int(
                round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))
            ),
            int(
                round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))
            ),
        )
        cv2.line(
            result,
            tuple(ex_point_1),
            tuple(point_1),
            4096.0,
            1,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        ex_point_2 = (
            int(
                round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))
            ),
            int(
                round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))
            ),
        )
        cv2.line(
            result,
            tuple(ex_point_2),
            tuple(point_2),
            4096.0,
            1,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        return ex_point_1, ex_point_2


PROCESSOR_CLS = [
    MakeICDARData,
    MakeSegDetectionData,
    MakeBorderMap,
]
