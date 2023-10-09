# coding: utf-8
# Copyright (C) 2021-2023, [Breezedeus](https://github.com/breezedeus).
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
import hashlib
import requests
from pathlib import Path
from typing import Tuple, Union, List, Dict, Any
import logging
import platform
import zipfile
from functools import cmp_to_key
import shutil
import tempfile

from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from huggingface_hub import hf_hub_download

from ..consts import MODEL_VERSION, MODEL_CONFIGS, AVAILABLE_MODELS


fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' 'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()


def set_logger(log_file=None, log_level=logging.INFO, log_file_level=logging.NOTSET):
    """
    Example:
        >>> set_logger(log_file)
        >>> logger.info("abc'")
    """
    log_format = logging.Formatter(fmt)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        if not Path(log_file).parent.exists():
            os.makedirs(Path(log_file).parent)
        if isinstance(log_file, Path):
            log_file = str(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def model_fn_prefix(backbone, epoch):
    return 'cnstd-v%s-%s-%04d.params' % (MODEL_VERSION, backbone, epoch)


def check_context(context):
    if isinstance(context, str):
        return any([ctx in context.lower() for ctx in ('gpu', 'cpu', 'cuda')])
    if isinstance(context, list):
        if len(context) < 1:
            return False
        return all(isinstance(ctx, torch.device) for ctx in context)
    return isinstance(context, torch.device)


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'cnstd')
    else:
        return os.path.join(os.path.expanduser("~"), '.cnstd')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('CNSTD_HOME', data_dir_default())


def check_model_name(model_name):
    assert model_name in MODEL_CONFIGS


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, download_source='CN', overwrite=False, sha1_hash=None):
    """Download a given URL
    Parameters
    ----------
    url : dict, url for downloading the model, with keys:
            repo_id, subfolder, filename
    path : str, optional
        Destination path to store downloaded file. By default, stores to the
        current directory with same name as in url.
    download_source: which OSS source will be used, 'CN' or 'HF'
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url['filename']
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url['filename'])
        else:
            fname = path

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if download_source == 'CN' and 'cn_oss' in url:
            oss_url = url['cn_oss'] + url['filename']
            logger.info('Downloading %s from %s...' % (fname, oss_url))
            r = requests.get(oss_url, stream=True)
            if r.status_code != 200:
                raise RuntimeError("Failed downloading url %s" % oss_url)
            total_length = r.headers.get('content-length')
            with open(fname, 'wb') as f:
                if total_length is None:  # no content length header
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                else:
                    total_length = int(total_length)
                    for chunk in tqdm(
                            r.iter_content(chunk_size=1024),
                            total=int(total_length / 1024.0 + 0.5),
                            unit='KB',
                            unit_scale=False,
                            dynamic_ncols=True,
                    ):
                        f.write(chunk)
        else:
            HF_TOKEN = os.environ.get('HF_TOKEN')
            logger.info('Downloading %s from HF Repo %s...' % (fname, url["repo_id"]))
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = hf_hub_download(
                    repo_id=url["repo_id"],
                    subfolder=url["subfolder"],
                    filename=url["filename"],
                    repo_type="model",
                    cache_dir=tmp_dir,
                    token=HF_TOKEN,
                )
                shutil.copy2(local_path, fname)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                'File {} is downloaded but the content hash does not match. '
                'The repo may be outdated or download may be incomplete. '
                'If the "repo_url" is overridden, consider switching to '
                'the default repo.'.format(fname)
            )
    return fname


class ModelDownloadingError(Exception):
    pass


def get_model_file(url, model_dir, download_source='CN'):
    r"""Return location for the downloaded models on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    url : dict, url for downloading the model, with keys:
            repo_id, subfolder, filename
    model_dir : str, default $CNSTD_HOME
        Location for keeping the model parameters.
    download_source : which OSS source will be used, 'CN' or 'HF'

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    model_dir = os.path.expanduser(model_dir)
    par_dir = os.path.dirname(model_dir)
    os.makedirs(par_dir, exist_ok=True)

    zip_file_path = os.path.join(par_dir, url['filename'])
    if not os.path.exists(zip_file_path):
        try:
            download(url, path=zip_file_path, download_source=download_source, overwrite=True)
        except Exception as e:
            logger.error(e)
            message = f'Failed to download model: {url["filename"]}.'
            message += '\n\tPlease open your VPN and try again. \n\t' \
                       'If this error persists, please follow the instruction at ' \
                       '[CnSTD/CnOCR Doc](https://www.breezedeus.com/cnocr) to manually download the model files.'
            raise ModelDownloadingError(message)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(par_dir)
    os.remove(zip_file_path)

    return model_dir


def read_charset(charset_fp):
    alphabet = []
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    if len(alphabet) != len(inv_alph_dict):
        from collections import Counter

        repeated = Counter(alphabet).most_common(len(alphabet) - len(inv_alph_dict))
        raise ValueError('repeated chars in vocab: %s' % repeated)

    return alphabet, inv_alph_dict


RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])


def normalize_img_array(img: np.ndarray, dtype='float32'):
    """ rescale to [-1.0, 1.0]
    :param img: RGB style, [H, W, 3] or [3, H, W]
    :param dtype: resulting dtype
    """

    img = img.astype(dtype)
    img_mean = RGB_MEAN.reshape(3, 1, 1) if img.shape[0] == 3 else RGB_MEAN
    img -= img_mean
    img = img / 255.0
    # img -= np.array((0.485, 0.456, 0.406))
    # img /= np.array((0.229, 0.224, 0.225))
    return img


def restore_img(img):
    """

    Args:
        img: [H, W, 3] or [3, H, W]

    Returns: [H, W, 3] or [3, H ,W]

    """
    img_mean = RGB_MEAN.reshape(3, 1, 1) if img.shape[0] == 3 else RGB_MEAN
    img = img * 255 + img_mean
    return img.clip(0, 255).astype(np.uint8)


def read_img(img_fp) -> Image.Image:
    img = Image.open(img_fp)
    img = ImageOps.exif_transpose(img).convert('RGB')  # 识别旋转后的图片（pillow不会自动识别）
    return img


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """

    Args:
        img: an Image.Image from `read_img()`

    Returns: np.ndarray, RGB-style, with shape: [3, H, W], scale: [0, 255], dtype: float32

    """
    return np.asarray(img.convert('RGB'), dtype='float32').transpose((2, 0, 1))


def transform_rbbox_to_bbox(x, y, w, h, alpha):
    points = cv2.boxPoints(((x, y), (w, h), alpha))
    return np.array(sort_box_points(points))


def sort_box_points(points):
    """

    Args:
        points (): shape of [4, 2]

    Returns:

    """
    points = sorted(list(points), key=lambda x: x[0])
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box


def _compare_box(box1, box2, key):
    # 从上到下，从左到右
    # box1, box2 to: [xmin, ymin, xmax, ymax]
    box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
    box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]

    def y_iou():
        # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
        # 判断是否有交集
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return 0
        # 计算交集的高度
        y_min = max(box1[1], box2[1])
        y_max = min(box1[3], box2[3])
        return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))

    if y_iou() > 0.5:
        return box1[0] - box2[0]
    else:
        return box1[1] - box2[1]


def sort_boxes(
    dt_boxes: List[Union[Dict[str, Any], Tuple[np.ndarray, float]]],
    key: Union[str, int] = 'box',
) -> List[Union[Dict[str, Any], Tuple[np.ndarray, float]]]:
    """
    Sort resulting boxes in order from top to bottom, left to right
    args:
        dt_boxes(array): list of dict or tuple, box with shape [4, 2]
    return:
        sorted boxes(array): list of dict or tuple, box with shape [4, 2]
    """
    _boxes = sorted(dt_boxes, key=cmp_to_key(lambda x, y: _compare_box(x, y, key)))
    return _boxes


def imread(img_fp) -> np.ndarray:
    """
    返回RGB格式的numpy数组
    Args:
        img_fp:

    Returns:
        RGB format ndarray: [C, H, W]

    """
    try:
        im = cv2.imread(img_fp, cv2.IMREAD_COLOR)  # res: color BGR, shape: [H, W, C]
        im = cv2.cvtColor(im.astype('float32'), cv2.COLOR_BGR2RGB)
    except:
        im = np.asarray(Image.open(img_fp).convert('RGB'))
    return im.transpose((2, 0, 1))


def imsave(image: np.ndarray, fp, normalized=True):
    """

    Args:
        image: [H, W, C]
        fp:
        normalized:

    Returns:

    """
    if normalized:
        image = restore_img(image)
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fp, image)


def get_resized_ratio(
    ori_hw: Tuple[int, int], target_hw: Tuple[int, int], preserve_aspect_ratio: bool
) -> Tuple[float, float]:
    """
    get height and weight ratios when resizing an image from original height and weight to target height and weight
    Args:
        ori_hw:
        target_hw:
        preserve_aspect_ratio:

    Returns:

    """
    ori_h, ori_w = ori_hw
    new_h, new_w = target_hw

    if preserve_aspect_ratio:
        target_ratio = new_h / new_w
        actual_ratio = ori_h / ori_w
        if actual_ratio > target_ratio:
            ratio = new_h / ori_h
        else:
            ratio = new_w / ori_w
        return ratio, ratio
    else:
        return new_h / ori_h, new_w / ori_w


def get_resized_shape(
    ori_hw: Tuple[int, int],
    target_hw: Union[int, Tuple[int, int]],
    preserve_aspect_ratio: bool,
    divided_by: int = 32,
) -> Tuple[int, int]:
    """
    获得满足要求的新图片尺寸： (height, weight)
    Args:
        ori_hw ():
        target_hw ():
        preserve_aspect_ratio ():
        divided_by (int): `>0` 表示最终尺寸能被此数整除，`<0` 则表示无此要求。默认为 `32`

    Returns:

    """
    if isinstance(target_hw, int):
        target_hw = (target_hw, target_hw)
    ratio = get_resized_ratio(
        ori_hw, target_hw, preserve_aspect_ratio=preserve_aspect_ratio
    )
    new_hw = (max(1, int(ori_hw[0] * ratio[0])), max(1, int(ori_hw[1] * ratio[1])))

    if divided_by <= 0:
        return new_hw

    def calibrate(ori):
        return max(int(round(ori / divided_by) * divided_by), divided_by)

    return calibrate(new_hw[0]), calibrate(new_hw[1])


def load_model_params(model, param_fp, device='cpu'):
    checkpoint = torch.load(param_fp, map_location=device)
    state_dict = checkpoint['state_dict']
    if all([param_name.startswith('model.') for param_name in state_dict.keys()]):
        # 表示导入的模型是通过 PlTrainer 训练出的 WrapperLightningModule，对其进行转化
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.split('.', maxsplit=1)[1]] = v
    model.load_state_dict(state_dict)
    return model


def draw_polygons(
    image: Union[np.ndarray, Image.Image],
    polygons: List[np.ndarray],
    ignore_tags: List[bool],
):
    """

    Args:
        image: [H, W, 3] for np.ndarray
        polygons:
        ignore_tags:

    Returns:

    """
    image = image.copy()
    if isinstance(image, Image.Image):
        image = pil_to_numpy(image).transpose((1, 2, 0))  # [H, W, 3]
    image = np.ascontiguousarray(image)
    for i in range(len(polygons)):
        polygon = polygons[i].reshape(-1, 2).astype(np.int32)
        ignore = ignore_tags[i]
        if ignore:
            color = (255, 0, 0)  # depict ignorable polygons in red
        else:
            color = (0, 0, 255)  # depict polygons in blue
        cv2.polylines(image, [polygon], True, color, 1)
    return image


def plot_for_debugging(rotated_img, one_out, box_score_thresh, prefix_fp):
    import matplotlib.pyplot as plt
    import math

    rotated_img = rotated_img.copy()
    crops = [info['cropped_img'] for info in one_out]
    logger.info('%d boxes are found' % len(crops))
    ncols = 3
    nrows = math.ceil(len(crops) / ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i, axi in enumerate(ax.flat):
        if i >= len(crops):
            break
        axi.imshow(crops[i])
    crop_fp = '%s-crops.png' % prefix_fp
    plt.savefig(crop_fp)
    logger.info('cropped results are save to file %s' % crop_fp)

    for info in one_out:
        box, score = info['box'], info['score']
        if score < box_score_thresh:  # score < 0.5
            continue
        box = box.astype(int).reshape(-1, 2)
        cv2.polylines(rotated_img, [box], True, color=(255, 0, 0), thickness=2)
    result_fp = '%s-result.png' % prefix_fp
    imsave(rotated_img, result_fp, normalized=False)
    logger.info('boxes results are save to file %s' % result_fp)
