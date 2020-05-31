# coding: utf-8
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
import time
import logging

import cv2
import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms

from .consts import MODEL_VERSION, AVAILABLE_MODELS
from .model.net import PSENet
from .model.pse import pse
from .utils import (
    data_dir,
    check_model_name,
    model_fn_prefix,
    get_model_file,
    imread,
    normalize_img_array,
)

logger = logging.getLogger(__name__)


class CnStd(object):
    """
    场景文字检测器（Scene Text Detection）。虽然名字中有个"Cn"（Chinese），但其实也可以轻松识别英文的。
    """

    def __init__(
        self, model_name='mobilenetv3', model_epoch=None, root=data_dir(), context='cpu'
    ):
        check_model_name(model_name)
        self._model_name = model_name
        self._model_epoch = model_epoch or AVAILABLE_MODELS[model_name][0]
        self._model_file_name = model_fn_prefix(model_name, self._model_epoch)

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, self._model_name)
        self._assert_and_prepare_model_files()
        model_fp = os.path.join(self._model_dir, self._model_file_name)

        if isinstance(context, mx.Context):
            self._context = context
        elif isinstance(context, str):
            self._context = mx.gpu() if context.lower() == 'gpu' else mx.cpu()
        else:
            self._context = mx.cpu()
        logger.info('CnStd is initialized, with context {}'.format(self._context))

        self._model = restore_model(
            self._model_name, model_fp, n_kernel=3, ctx=self._context
        )
        self._trans = transforms.Compose([transforms.ToTensor(),])
        self.seg_maps = None

    def _assert_and_prepare_model_files(self):
        model_dir = self._model_dir
        model_files = [self._model_file_name]
        file_prepared = True
        for f in model_files:
            f = os.path.join(model_dir, f)
            if not os.path.exists(f):
                file_prepared = False
                logger.warning('can not find file %s', f)
                break

        if file_prepared:
            return

        get_model_file(model_dir)

    def detect(self, img_fp, max_size=768, pse_threshold=0.45, pse_min_area=100):
        """
        检测图片中的文本。
        Args:
            img_fp: image file path; or color image mx.nd.NDArray or np.ndarray,
            with shape (height, width, 3), and the channels should be RGB formatted.
            max_size: 如果图片的长边超过这个值，就把图片等比例压缩到长边等于这个size
            pse_threshold: pse中的阈值；越低会导致识别出的文本框越大；反之越小
            pse_min_area: 面积大小低于此值的框会被去掉。所以此值越小，识别出的框可能越多

        Returns: List(Dict), 每个元素存储了检测出的一个框的信息，使用词典记录，包括以下几个值：
                    'box'：检测出的文字对应的矩形框四个点的坐标（第一列是宽度方向，第二列是高度方向）；
                           np.ndarray类型，shape==(4, 2)；
                    'score'：得分；float类型；
                    'croppped_img'：对应'box'中的图片patch（RGB格式），会把倾斜的图片旋转为水平。
                           np.ndarray类型，shape==(width, height, 3)；

          示例:
            [{'box': array([[416,  77],
                            [486,  13],
                            [800, 325],
                            [730, 390]], dtype=int32),
              'score': 1.0, 'cropped_img': array([[[25, 20, 24],
                                                   [26, 21, 25],
                                                   [25, 20, 24],
                                                   ...,
                                                   [11, 11, 13],
                                                   [11, 11, 13],
                                                   [11, 11, 13]]], dtype=uint8)},
             ...
            ]

        """
        if isinstance(img_fp, str):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = imread(img_fp)
        elif isinstance(img_fp, mx.nd.NDArray):
            img = img_fp.asnumpy()
        elif isinstance(img_fp, np.ndarray):
            img = img_fp
        else:
            raise TypeError('Inappropriate argument type.')
        if min(img.shape[0], img.shape[1]) < 2:
            return []
        logger.debug("processing image with shape {}".format(img.shape))

        resize_img, (ratio_h, ratio_w) = resize_image(img, max_side_len=max_size)

        h, w, _ = resize_img.shape
        resize_img = normalize_img_array(resize_img)
        im_res = mx.nd.array(resize_img)
        im_res = self._trans(im_res)

        t1 = time.time()
        self.seg_maps = self._model(
            im_res.expand_dims(axis=0).as_in_context(self._context)
        ).asnumpy()
        t2 = time.time()
        boxes, scores, rects = detect_pse(
            self.seg_maps,
            threshold=pse_threshold,
            threshold_k=pse_threshold,
            boxes_thres=0.01,
            min_area=pse_min_area,
        )
        t3 = time.time()
        logger.debug(
            "\tfinished, time costs: psenet pred {}, pse {}, text_boxes: {}".format(
                t2 - t1, t3 - t2, len(boxes)
            )
        )

        if len(boxes) == 0:
            return []

        boxes = boxes.reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        boxes = boxes.astype('int32')

        cropped_imgs = []
        for idx, rect in enumerate(rects):
            # import pdb; pdb.set_trace()
            # cv2.drawContours(img, [np.int0(bboxes[idx])], 0, (0, 0, 255), 3)
            # cv2.imwrite('img_box.jpg', img)
            rect = resize_rect(rect, 1.0 / ratio_w, 1.0 / ratio_h)
            cropped_img = crop_rect(img, rect, alph=0.05)
            cropped_imgs.append(cropped_img)
            # cv2.imwrite("img_crop_rot%d.jpg" % idx, cropped_img)

        names = ('box', 'score', 'cropped_img')
        final_res = []
        for one_info in zip(boxes, scores, cropped_imgs):
            one_dict = dict(zip(names, one_info))
            one_dict['box'] = sort_poly(one_dict['box'])
            final_res.append(one_dict)
        return final_res


def restore_model(backbone, ckpt_path, n_kernel, ctx):
    """
    Restore model and get runtime session, input, output
    Args:
        - ckpt_path: the path to checkpoint file
        - n_kernel: [kernel_map, score_map]
    """

    net = PSENet(base_net_name=backbone, num_kernels=n_kernel, ctx=ctx)
    net.load_parameters(ckpt_path)

    return net


def resize_image(im, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / max(resize_h, resize_w)
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else max(1, resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else max(1, resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def mask_to_boxes_pse(result_map, score_map, min_score=0.5, min_area=200, scale=4.0):
    """
    Generate boxes from mask
    Args:
        - result_map: fusion from kernel maps
        - score_map: text_region
        - min_score: the threshold to filter box that lower than min_score
        - min_area: filter box whose area is smaller than min_area
        - scale: ratio about input and output of network
    """
    label = result_map
    label_num = np.max(label) + 1
    boxes = []
    scores = []
    rects = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area / (scale * scale):
            continue

        score_i = np.mean(score_map[label == i])
        if score_i < min_score:
            continue

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect) * scale
        boxes.append(box.reshape(-1))
        scores.append(score_i)

        rect = resize_rect(rect, scale, scale)
        rects.append(rect)
    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), rects


def detect_pse(
    seg_maps, threshold=0.5, threshold_k=0.55, boxes_thres=0.01, min_area=100
):
    """
    poster with pse
    """
    seg_maps = seg_maps[0, :, :, :]
    mask = np.where(seg_maps[0, :, :] > threshold, 1.0, 0.0)
    seg_maps = seg_maps * mask > threshold_k

    result_map = pse(seg_maps, 5)
    boxes, scores, rects = mask_to_boxes_pse(
        result_map, seg_maps[0, :, :], min_score=boxes_thres, min_area=min_area
    )
    return boxes, scores, rects


def crop_rect(img, rect, alph=0.05):
    """
    adapted from https://github.com/ouyanghuiyu/chineseocr_lite/blob/e959b6dbf3/utils.py
    从图片中按框截取出图片patch。
    """
    center, sizes, angle = rect[0], rect[1], rect[2]
    sizes = (int(sizes[0] * (1 + alph)), int(sizes[1] + sizes[0] * alph))
    center = (int(center[0]), int(center[1]))

    if 1.5 * sizes[0] < sizes[1]:
        sizes = (sizes[1], sizes[0])
        angle += 90

    height, width = img.shape[0], img.shape[1]
    # 先把中心点平移到图片中心，然后再旋转就不会截断图片了
    img = translate(img, width // 2 - center[0], height // 2 - center[1])
    center = (width // 2, height // 2)

    # FIXME 如果遇到一个贯穿整个图片对角线的文字行，旋转还是会有边缘被截断的情况
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, sizes, center)
    # cv2.imwrite("img_translate.jpg", img)
    # cv2.imwrite("img_rot.jpg", img_rot)
    # cv2.imwrite("img_crop_rot.jpg", img_crop)
    # import pdb; pdb.set_trace()
    return img_crop


def translate(image, x, y):
    """from https://www.programcreek.com/python/example/89459/cv2.getRotationMatrix2D:
        Example 29
    """
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 返回转换后的图像
    return shifted


def resize_rect(rect, w_scale, h_scale):
    # FIXME 如果w_scale和h_scale不同，angle其实也要对应修正
    center, sizes, angle = rect
    center = (center[0] * w_scale, center[1] * h_scale)
    sizes = (sizes[0] * w_scale, sizes[1] * h_scale)
    rect = (center, sizes, angle)
    return rect


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
