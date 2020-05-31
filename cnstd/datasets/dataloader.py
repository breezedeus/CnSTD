# coding=utf-8
import os
import logging

import cv2
import mxnet as mx
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.dataset import Dataset

from ..utils import imread, normalize_img_array
from .util import parse_lines, process_data

logger = logging.getLogger(__name__)


class STRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train_idx_fp,
        strides=4,
        input_size=(640, 640),
        num_kernels=6,
        debug=False,
    ):
        super(STRDataset, self).__init__()
        img_label_pairs = read_idx_file(train_idx_fp)
        self.img_label_pairs = [
            (os.path.join(root_dir, img_fp), os.path.join(root_dir, gt_fp))
            for img_fp, gt_fp in img_label_pairs
        ]

        self.length = len(self.img_label_pairs)
        self.input_size = input_size
        self.strides = strides
        self.num_kernel = num_kernels
        self.debug = debug
        self.trans = transforms.Compose(
            [
                # transforms.RandomColorJitter(brightness = 32.0 / 255, saturation = 0.5),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        logger.info('data size: {}'.format(len(self)))

    def __getitem__(self, item):
        img_fp, label_fp = self.img_label_pairs[item]
        text_polys, text_tags = parse_lines(label_fp)
        im = imread(img_fp)
        if len(im.shape) != 3 or im.shape[2] != 3:
            logger.warning('bad image: {}, with shape {}'.format(img_fp, im.shape))

        # gt_kernels 是从大到小的，与论文中使用的下标刚好相反
        image, gt_text, gt_kernels, training_mask = process_data(
            im, text_polys, text_tags, self.num_kernel
        )
        if self.debug:
            im_show = np.concatenate(
                [
                    gt_text * 255,
                    gt_kernels[0, :, :] * 255,
                    gt_kernels[1, :, :] * 255,
                    training_mask * 255,
                ],
                axis=1,
            )
            cv2.imshow('img', image)
            cv2.imshow('score_map', im_show)
            cv2.waitKey()
        image = normalize_img_array(image)
        image = mx.nd.array(image)
        gt_text = mx.nd.array(gt_text, dtype=np.float32)
        kernal_map = mx.nd.array(gt_kernels, dtype=np.float32)
        training_mask = mx.nd.array(training_mask, dtype=np.float32)
        trans_image = self.trans(image)
        return (
            trans_image,
            gt_text,
            kernal_map,
            training_mask,
            transforms.ToTensor()(image),
        )

    def __len__(self):
        return self.length


def read_idx_file(idx_fp):
    img_label_pairs = []
    with open(idx_fp) as f:
        for line in f:
            img_fp, gt_fp = line.strip().split('\t')
            img_label_pairs.append((img_fp, gt_fp))
    return img_label_pairs
