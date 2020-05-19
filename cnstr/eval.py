# coding=utf-8
import os
import sys
import glob
import time
import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

# from pse import pse
from .postprocess.pse_poster import pse
from .utils import imread
from .model.net import PSENet


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
        ratio = (
            float(max_side_len) / resize_h
            if resize_h > resize_w
            else float(max_side_len) / resize_w
        )
    else:
        ratio = 1.0
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
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
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area / (scale * scale):
            continue

        score_i = np.mean(score_map[label == i])
        if score_i < min_score:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect) * scale
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))
        scores.append(score_i)
    return np.array(bboxes, dtype=np.float32), np.array(scores, dtype=np.float32)


def detect_pse(seg_maps, threshold=0.5, threshold_k=0.55, boxes_thres=0.01):
    """
    poster with pse
    """
    seg_maps = seg_maps[0, :, :, :]
    mask = np.where(seg_maps[0, :, :] > threshold, 1.0, 0.0)
    seg_maps = seg_maps * mask > threshold_k

    result_map = pse(seg_maps, 5)
    bboxes, scores = mask_to_boxes_pse(
        result_map, seg_maps[0, :, :], min_score=boxes_thres
    )
    return bboxes, scores


def restore_model(ckpt_path, n_kernel, ctx):
    """
    Restore model and get runtime session, input, output
    Args:
        - ckpt_path: the path to checkpoint file
        - n_kernel: [kernel_map, score_map]
    """

    net = PSENet(num_kernels=n_kernel, ctx=ctx)
    net.load_parameters(ckpt_path)

    return net


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


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def evaluate(image_dir, ckpt_path, output_dir, ctx=None):
    # restore model
    net = restore_model(ckpt_path, n_kernel=3, ctx=ctx)
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # process image
    imglst = glob.glob1(image_dir, "*g")
    for item in imglst:
        im_name = os.path.join(image_dir, item)
        img = imread(im_name)
        resize_img, (ratio_h, ratio_w) = resize_image(img, max_side_len=784)

        h, w, _ = resize_img.shape
        im_res = mx.nd.array(resize_img)
        im_res = trans(im_res)

        t1 = time.time()
        seg_maps = net(im_res.expand_dims(axis=0)).asnumpy()
        t2 = time.time()
        bboxes, scores = detect_pse(seg_maps, threshold=0.5)
        t3 = time.time()

        print("net: {}, nms: {}, text_boxes: {}".format(t2 - t1, t3 - t2, len(bboxes)))
        if len(bboxes) > 0:
            bboxes = bboxes.reshape((-1, 4, 2))
            bboxes[:, :, 0] /= ratio_w
            bboxes[:, :, 1] /= ratio_h

            # save result
            out_text_name = os.path.join(output_dir, item[:-4] + '.txt')
            out_img_name = os.path.join(output_dir, item)
            out_fusion_img_name = os.path.join(output_dir, "fusion_" + item)
            with open(out_text_name, 'w') as f:
                for box in bboxes:
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
                        color=(255, 255, 0),
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

            fusion_img = weighted_fusion(seg_maps, img)
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
