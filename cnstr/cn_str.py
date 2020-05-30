# coding=utf-8
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
from .postprocess.pse_poster import pse
from .utils import (
    data_dir,
    check_model_name,
    model_fn_prefix,
    get_model_file,
    imread,
    normalize_img_array,
)

logger = logging.getLogger(__name__)


class CnStr(object):
    """
    场景文字识别器（Scene Text Recognition）。虽然名字中有个"Cn"（Chinese），但其实也可以轻松识别英文的。
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
        logger.info('CnStr is initialized, with context {}'.format(self._context))

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

    def recognize(self, img_fp, max_size=768, pse_threshold=0.45, pse_min_area=100):
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
            final_res.append(dict(zip(names, one_info)))
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
    bboxes = []
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
        bbox = cv2.boxPoints(rect) * scale
        bboxes.append(bbox.reshape(-1))
        scores.append(score_i)

        rect = resize_rect(rect, scale, scale)
        rects.append(rect)
    return np.array(bboxes, dtype=np.float32), np.array(scores, dtype=np.float32), rects


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
    bboxes, scores, rects = mask_to_boxes_pse(
        result_map, seg_maps[0, :, :], min_score=boxes_thres, min_area=min_area
    )
    return bboxes, scores, rects


def crop_rect(img, rect, alph=0.05):
    center, sizes, angle = rect[0], rect[1], rect[2]
    sizes = (int(sizes[0] * (1 + alph)), int(sizes[1] + sizes[0] * alph))
    center = (int(center[0]), int(center[1]))

    if 1.5 * sizes[0] < sizes[1]:
        sizes = (sizes[1], sizes[0])
        angle += 90

    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, sizes, center)
    # cv2.imwrite("img_rot.jpg", img_rot)
    # cv2.imwrite("img_crop_rot.jpg", img_crop)
    # import pdb; pdb.set_trace()
    # img_crop = Image.fromarray(img_crop)
    return img_crop


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
