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
import os
import sys
import glob
import logging
import cv2
import numpy as np

from .utils import imread
from .cn_std import CnStd

logger = logging.getLogger(__name__)


def weighted_fusion(seg_maps, image):
    h, w, _ = image.shape
    seg_maps = np.squeeze(seg_maps)
    for i, seg_map in enumerate(seg_maps):
        # check file exits
        seg_map = np.expand_dims(seg_map, axis=2)
        seg_map_3c = np.repeat(seg_map, 3, 2) * 255
        seg_map_3c = cv2.resize(
            seg_map_3c, dsize=(w, h), interpolation=cv2.INTER_LINEAR
        )
        att_im = cv2.addWeighted(seg_map_3c.astype(np.uint8), 0.5, image, 0.5, 0.0)
        save_img = att_im if i == 0 else np.concatenate((save_img, att_im), 1)
    return save_img


def evaluate(
    backbone,
    model_root_dir,
    model_epoch,
    image_dir,
    output_dir,
    max_size,
    pse_threshold,
    pse_min_area,
    ctx=None,
):
    # process image
    if os.path.isfile(image_dir):
        imglst = [image_dir]
    elif os.path.isdir(image_dir):
        imglst = glob.glob1(image_dir, "*g")
        imglst = [os.path.join(image_dir, fn) for fn in imglst]
    else:
        logger.error('param "image_dir": %s is neither a file or a dir' % image_dir)
        raise TypeError(
            'param "image_dir": %s is neither a file or a dir' % image_dir
        )

    # restore model
    cn_str = CnStd(
        model_name=backbone, model_epoch=model_epoch, root=model_root_dir, context=ctx
    )

    for im_name in imglst:
        item = os.path.basename(im_name)
        out_fusion_img_name = os.path.join(output_dir, "fusion_" + item)
        if os.path.exists(out_fusion_img_name):
            continue
        logger.info("processing image {}".format(item))
        box_info_list = cn_str.detect(im_name, max_size, pse_threshold, pse_min_area)

        img = imread(im_name)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(box_info_list) > 0:
            # save result
            out_text_name = os.path.join(output_dir, item[:-4] + '.txt')
            with open(out_text_name, 'w') as f:
                for idx, box_info in enumerate(box_info_list):
                    box = box_info['box']
                    if (
                        np.linalg.norm(box[0] - box[1]) < 5
                        or np.linalg.norm(box[3] - box[0]) < 5
                    ):
                        continue
                    cv2.polylines(
                        img,
                        [box.astype(np.int32).reshape((-1, 1, 2))],
                        True,
                        color=(0, 0, 255),
                        thickness=1,
                    )
                    f.write(
                        '{},{},{},{},{},{},{},{}\r\n'.format(
                            box[0, 0],
                            box[0, 1],
                            box[1, 0],
                            box[1, 1],
                            box[2, 0],
                            box[2, 1],
                            box[3, 0],
                            box[3, 1],
                        )
                    )
                    cropped_img = box_info['cropped_img']
                    cropped_img_name = os.path.join(
                        output_dir, "fusion_" + item + "_crop%d.jpg" % idx
                    )
                    cv2.imwrite(
                        cropped_img_name, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                    )

            fusion_img = weighted_fusion(cn_str.seg_maps, img)
            fusion_img = np.concatenate((fusion_img, img), 1)
            cv2.imwrite(out_fusion_img_name, fusion_img)


if __name__ == "__main__":
    image_dir = sys.argv[1]
    ckpt_path = sys.argv[2]
    output_dir = sys.argv[3]
    gpu_list = sys.argv[4]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    evaluate(image_dir, ckpt_path, output_dir, gpu_list)
