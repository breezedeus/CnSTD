# coding=utf-8
import os
import sys
import glob
import time
import logging
import cv2
import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms

from .cn_str import restore_model, resize_image, detect_pse, sort_poly
from .utils import imread, normalize_img_array
from .cn_str import CnStr

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
    model_epoch,
    image_dir,
    output_dir,
    max_size,
    pse_threshold,
    pse_min_area,
    ctx=None,
):
    # restore model
    cn_str = CnStr(model_name=backbone, model_epoch=model_epoch, context=ctx)

    # process image
    imglst = glob.glob1(image_dir, "*g")
    for item in imglst:
        im_name = os.path.join(image_dir, item)
        out_fusion_img_name = os.path.join(output_dir, "fusion_" + item)
        if os.path.exists(out_fusion_img_name):
            continue
        logger.info("processing image {}".format(item))
        bboxe_score_list = cn_str.recognize(
            im_name, max_size, pse_threshold, pse_min_area
        )

        img = imread(im_name)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(bboxe_score_list) > 0:
            # save result
            out_text_name = os.path.join(output_dir, item[:-4] + '.txt')
            out_img_name = os.path.join(output_dir, item)
            with open(out_text_name, 'w') as f:
                for box, score in bboxe_score_list:
                    box = sort_poly(box.astype(np.int32))
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

            fusion_img = weighted_fusion(cn_str.seg_maps, img)
            fusion_img = np.concatenate((fusion_img, img), 1)
            # cv2.imwrite(out_img_name, img)
            cv2.imwrite(out_fusion_img_name, fusion_img)


if __name__ == "__main__":
    image_dir = sys.argv[1]
    ckpt_path = sys.argv[2]
    output_dir = sys.argv[3]
    gpu_list = sys.argv[4]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    evaluate(image_dir, ckpt_path, output_dir, gpu_list)
