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
# Credits: adapted from https://github.com/PaddlePaddle/PaddleOCR

import os
import time
from copy import deepcopy
from typing import Union, Optional, Any, List, Dict, Tuple
from pathlib import Path
import logging
import json

from PIL import Image
import cv2
import numpy as np

from .consts import PP_SPACE
from ..consts import MODEL_VERSION, AVAILABLE_MODELS, DOWNLOAD_SOURCE
from ..utils import data_dir, get_model_file, sort_boxes, get_resized_shape
from .utility import (
    get_image_file_list,
    check_and_read_gif,
    create_predictor,
    parse_args,
    draw_text_det_res,
    get_rotate_crop_image,
)
from .opt_utils import transform, create_operators
from .postprocess import build_post_process
from .img_operators import DetResizeForTest

logger = logging.getLogger(__name__)


class PPDetector(object):
    def __init__(
        self,
        model_name: str = 'ch_PP-OCRv3',
        *,
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        bin_thresh: float = 0.3,
        box_thresh: float = 0.6,
        limit_type='max',
        unclip_ratio=1.5,
        use_dilation=False,
        det_db_score_mode='fast',
        **kwargs,
    ):
        self._model_name = model_name
        self._model_backend = 'onnx'

        self._assert_and_prepare_model_files(model_fp, root)

        postprocess_params = {
            'name': 'DBPostProcess',
            "max_candidates": 1000,
            "unclip_ratio": unclip_ratio,
            "use_dilation": use_dilation,
            "score_mode": det_db_score_mode,
            "bin_thresh": bin_thresh,
            "box_thresh": box_thresh,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = create_predictor(self._model_fp, 'det', logger)

        limit_side_len = 960  # 这个值没用了， `detect()` 时用的是 `image_shape`
        pre_process_list = [
            {
                'DetResizeForTest': {
                    'limit_side_len': limit_side_len,
                    'limit_type': limit_type,
                }
            },
            {
                'NormalizeImage': {
                    'std': [0.229, 0.224, 0.225],
                    'mean': [0.485, 0.456, 0.406],
                    'scale': '1./255.',
                    'order': 'hwc',
                }
            },
            {'ToCHWImage': None},
            {'KeepKeys': {'keep_keys': ['image', 'shape']}},
        ]
        img_h, img_w = self.input_tensor.shape[2:]
        if img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {'DetResizeForTest': {'image_shape': [img_h, img_w]}}
        self.preprocess_op = create_operators(pre_process_list)

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, PP_SPACE)
        model_fp = os.path.join(self._model_dir, '%s_infer.onnx' % self._model_name)
        if not os.path.isfile(model_fp):
            logger.warning('can not find model file %s' % model_fp)
            if (self._model_name, self._model_backend) not in AVAILABLE_MODELS:
                raise NotImplementedError(
                    '%s is not a downloadable model'
                    % ((self._model_name, self._model_backend),)
                )
            url = AVAILABLE_MODELS.get_url(self._model_name, self._model_backend)

            get_model_file(url, self._model_dir, download_source=DOWNLOAD_SOURCE)  # download the .zip file and unzip

        self._model_fp = model_fp
        logger.info('use model: %s' % self._model_fp)

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape, min_box_size):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box, score in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if min(rect_width, rect_height) < min_box_size:
                continue
            dt_boxes_new.append((box, score))
        # dt_boxes = np.array(dt_boxes_new)
        return dt_boxes_new

    def detect(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        resized_shape: Union[int, Tuple[int, int]] = (768, 768),
        preserve_aspect_ratio: bool = True,
        box_score_thresh: float = 0.3,
        min_box_size: int = 4,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        outs = []
        for img in img_list:
            img = self._preprocess_images(img)
            outs.append(
                self.detect_one(
                    img,
                    resized_shape,
                    preserve_aspect_ratio,
                    box_score_thresh,
                    min_box_size,
                )
            )

        return outs

    def detect_one(
        self,
        img: np.ndarray,
        resized_shape: Union[int, Tuple[int, int]],
        preserve_aspect_ratio: bool,
        box_score_thresh: float = 0.6,
        min_box_size: int = 4,
    ):
        ori_im = img.copy()
        data = {'image': img}

        if isinstance(self.preprocess_op[0], DetResizeForTest):
            self.preprocess_op[0].resize_type = 1
            self.preprocess_op[0].image_shape = get_resized_shape(
                img.shape[:2], resized_shape, preserve_aspect_ratio, divided_by=32
            )
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        input_dict = {}
        input_dict[self.input_tensor.name] = img
        outputs = self.predictor.run(self.output_tensors, input_dict)

        preds = {'maps': outputs[0]}

        post_result = self.postprocess_op(
            preds, shape_list, box_thresh=box_score_thresh
        )
        dt_boxes = list(zip(post_result[0]['points'], post_result[0]['scores']))
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape, min_box_size)
        dt_boxes = sort_boxes(dt_boxes, key=0)

        detected_results = []
        for bno in range(len(dt_boxes)):
            box, score = dt_boxes[bno]
            img_crop = get_rotate_crop_image(ori_im, deepcopy(box))
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            detected_results.append(
                {'box': box, 'score': score, 'cropped_img': img_crop.astype('uint8')}
            )

        return dict(rotated_angle=0.0, detected_texts=detected_results)

    @classmethod
    def _preprocess_images(
        cls, img: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """

        Args:
            img ():

        Returns:
            BGR format ndarray: [H, W, 3]

        """
        if isinstance(img, (str, Path)):
            if not os.path.isfile(img):
                raise FileNotFoundError(img)
            return cv2.imread(img, cv2.IMREAD_COLOR)
        elif isinstance(img, Image.Image):
            img = np.asarray(img.convert('RGB'), dtype='float32')
        if isinstance(img, np.ndarray):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError('type %s is not supported now' % str(type(img)))


if __name__ == "__main__":
    args = parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = PPDetector(args)
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"

    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)

    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    save_results = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        st = time.time()
        dt_boxes, _ = text_detector.detect_one(img)
        elapse = time.time() - st
        if count > 0:
            total_time += elapse
        count += 1
        save_pred = (
            os.path.basename(image_file)
            + "\t"
            + str(json.dumps(np.array(dt_boxes).astype(np.int32).tolist()))
            + "\n"
        )
        save_results.append(save_pred)
        logger.info(save_pred)
        logger.info("The predict time of {}: {}".format(image_file, elapse))
        src_im = draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save, "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        logger.info("The visualized image saved in {}".format(img_path))

    with open(os.path.join(draw_img_save, "det_results.txt"), 'w') as f:
        f.writelines(save_results)
        f.close()
