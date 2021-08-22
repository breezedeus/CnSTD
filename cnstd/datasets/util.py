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

import numpy as np
import random
import cv2
from shapely.geometry import Polygon
import pyclipper
import os


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_crop(imgs, img_size):
    """

    :param imgs: 包含img和kernel
    :param img_size:
    :return:
    """
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i : i + th, j : j + tw, :]
        else:
            imgs[idx] = imgs[idx][i : i + th, j : j + tw]
    return imgs


def random_rotate(imgs):
    angle = np.random.uniform(-10, 10)
    cols = imgs[0].shape[1]
    rows = imgs[0].shape[0]

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    for idx in range(len(imgs)):
        imgs[idx] = cv2.warpAffine(imgs[idx], M, (cols, rows))

    return imgs


def poly_offset(img, poly, dis):
    subj_poly = np.array(poly)
    # Polygon(subj_poly).area, Polygon(subj_poly).length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(subj_poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(-1.0 * dis)
    ss = np.array(solution)
    cv2.fillPoly(img, ss.astype(np.int32), 1)
    return img


def cal_offset(poly, r, max_shr=20):
    area, length = Polygon(poly).area, Polygon(poly).length
    r = r * r
    d = area * (1 - r) / (length + 0.005) + 0.5
    d = min(int(d), max_shr)
    return d


def shrink_polys(img, polys, tags, mini_scale_ratio, num_kernels=6):
    h, w = img.shape[:2]
    f = lambda x: 1.0 - (1.0 - mini_scale_ratio) / (num_kernels - 1.0) * x
    r = [f(i + 1) for i in range(num_kernels)]
    training_mask = np.ones((h, w), dtype=np.float32)
    kernel_maps = np.zeros((h, w, num_kernels), dtype=np.float32)
    score_map = np.zeros((h, w), dtype=np.float32)
    for poly, tag in zip(polys, tags):
        poly = np.array(poly, dtype=np.float32).reshape((-1, 2))
        cv2.fillPoly(score_map, poly.astype(np.int32)[np.newaxis, :, :], 1)

    for i, val in enumerate(r):
        tmp_score_map = np.zeros((h, w), dtype=np.float32)
        for poly, tag in zip(polys, tags):
            poly = np.array(poly, dtype=np.float32).reshape((-1, 2))
            if tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            d = cal_offset(poly, val)
            tmp_score_map = poly_offset(tmp_score_map, poly, d)
        kernel_maps[:, :, i] = tmp_score_map
    # return [kernel_maps[:, :, i] for i in xrange(num_kernels)], training_mask
    return score_map, kernel_maps, training_mask


def parse_lines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # print(filename)
    text_polys = []
    text_tags = []
    if not os.path.exists(filename):
        return np.array(text_polys, dtype=np.float32)
    for line in lines:
        line = line.strip('\n').split(',')
        # print(line)
        label = line[-1]
        # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
        line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
        if 10 > len(line) > 7:
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        elif 7 > len(line) > 3:
            x0, y0, x1, y1 = list(map(float, line[:4]))
            text_polys.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        else:
            continue
        if label == '*' or label == '###':
            # 表示"Do Not Care"，可能是太小的文字或者其他东西
            text_tags.append(True)
        else:
            text_tags.append(False)
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_scale(img, text_polys, min_side=640):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > 1280.0:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    if text_polys is not None:
        text_polys *= scale
        text_polys = np.array(text_polys)

    h, w = img.shape[:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)

    if min(h, w) * scale < min_side:
        scale = (min_side + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    if text_polys is not None:
        text_polys *= scale
        text_polys = np.array(text_polys)
    return img, text_polys


def save_images(imgs):
    for i, item in enumerate(imgs):
        cv2.imwrite('img_{}.png'.format(i), item * 255)


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = Polygon(bbox).area
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)

    return np.array(shrinked_bboxes)


def process_data(image, bboxes, label, num_kernels=6):
    img, bboxes = random_scale(image, bboxes)

    gt_text = np.zeros(img.shape[0:2], dtype='uint8')
    training_mask = np.ones(img.shape[0:2], dtype='uint8')
    if bboxes.shape[0] > 0:
        for i in range(bboxes.shape[0]):
            cv2.drawContours(
                gt_text, bboxes[i][np.newaxis, :, :].astype(np.int32), -1, i + 1, -1
            )
            if label[i]:  # True，表示"Do Not Care"，可能是太小的文字或者其他东西
                cv2.drawContours(
                    training_mask,
                    bboxes[i][np.newaxis, :, :].astype(np.int32),
                    -1,
                    0,
                    -1,
                )

    gt_kernals = []
    f = lambda x: 1.0 - (1.0 - 0.5) / (num_kernels) * x
    rates = [f(i + 1) for i in range(num_kernels)]

    # from large kernel to small kernel
    for rate in rates:
        gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
        kernal_bboxes = shrink(bboxes, rate)
        for i in range(bboxes.shape[0]):
            cv2.drawContours(
                gt_kernal,
                kernal_bboxes[i][np.newaxis, :, :].astype(np.int32),
                -1,
                1,
                -1,
            )
        gt_kernals.append(gt_kernal)

    imgs = [img, gt_text, training_mask]
    imgs.extend(gt_kernals)

    imgs = random_horizontal_flip(imgs)
    imgs = random_rotate(imgs)
    imgs = random_crop(imgs, (640, 640))

    img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]

    gt_text[gt_text > 0] = 1
    gt_kernals = np.array(gt_kernals)

    return img, gt_text, gt_kernals, training_mask
