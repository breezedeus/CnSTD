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
import math
import logging
import traceback
from pathlib import Path
from typing import Union, Optional, Any, List, Dict

import cv2
import numpy as np

from ..consts import MODEL_VERSION, ANGLE_CLF_MODELS, ANGLE_CLF_SPACE
from ..utils import data_dir, get_model_file
from .postprocess import build_post_process
from .utility import (
    get_image_file_list,
    check_and_read_gif,
    create_predictor,
    parse_args,
)

logger = logging.getLogger(__name__)


class AngleClassifier(object):
    def __init__(
        self,
        model_name: str = 'ch_ppocr_mobile_v2.0_cls',
        *,
        model_fp: Optional[str] = None,
        clf_image_shape='3, 48, 192',
        clf_batch_num=6,
        clf_thresh=0.9,
        label_list=['0', '180'],
        root: Union[str, Path] = data_dir(),
    ):
        self._model_name = model_name
        self._model_backend = 'onnx'
        self.clf_image_shape = [int(v) for v in clf_image_shape.split(",")]
        self.clf_batch_num = clf_batch_num
        self.clf_thresh = clf_thresh

        self._assert_and_prepare_model_files(model_fp, root)

        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, _ = create_predictor(
            self._model_fp, 'cls', logger
        )

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        self._model_dir = os.path.join(root, MODEL_VERSION, ANGLE_CLF_SPACE)
        model_fp = os.path.join(self._model_dir, '%s_infer.onnx' % self._model_name)
        if not os.path.isfile(model_fp):
            logger.warning('can not find model file %s' % model_fp)
            if (self._model_name, self._model_backend) not in ANGLE_CLF_MODELS:
                raise NotImplementedError(
                    '%s is not a downloadable model'
                    % ((self._model_name, self._model_backend),)
                )
            url = ANGLE_CLF_MODELS[(self._model_name, self._model_backend)]['url']

            get_model_file(url, self._model_dir)  # download the .zip file and unzip

        self._model_fp = model_fp

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.clf_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.clf_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        """

        Args:
            img_list (list): each element with shape [H, W, 3], RGB-formated image

        Returns:
            img_list (list): rotated images, each element with shape [H, W, 3], RGB-formated image
            cls_res (list):

        """
        img_list = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img_list]

        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.clf_batch_num
        for beg_img_no in range(0, img_num, batch_num):

            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors, input_dict)
            prob_out = outputs[0]
            cls_result = self.postprocess_op(prob_out)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.clf_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )

        img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
        return img_list, cls_res


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_classifier = AngleClassifier(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        img_list, cls_res, predict_time = text_classifier(img_list)
    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    for ino in range(len(img_list)):
        logger.info(
            "Predicts of {}:{}".format(valid_image_file_list[ino], cls_res[ino])
        )


if __name__ == "__main__":
    main(parse_args())
