# coding: utf-8

import os
import logging
from pathlib import Path
from typing import Union, Optional, Any, List, Dict, Tuple

from PIL import Image
import cv2
import numpy as np
import torch
from torch import nn
from numpy import random

from ..consts import MODEL_VERSION, LAYOUT_SPACE, LAYOUT_MODELS
from ..utils import data_dir, get_model_file
from .yolo import Model
from .consts import CATEGORIES
from .common import Conv
from .datasets import letterbox
from .general import (
    check_img_size,
    non_max_suppression,
    xyxy24p,
    scale_coords,
)
from .torch_utils import select_device, time_synchronized
from .plots import plot_one_box

logger = logging.getLogger(__name__)


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


@torch.no_grad()
def attempt_load(
    model_fp, cfg_fp=Path(__file__).parent / 'yolov7-tiny.yaml', map_location=None
):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    inner_model = Model(cfg_fp, ch=3, nc=len(CATEGORIES), anchors=None).to(
        map_location
    )  # create
    state_dict = torch.load(model_fp, map_location=map_location)  # load
    inner_model.load_state_dict(state_dict)
    # inner_model.names = CATEGORIES

    model = Ensemble()
    model.append(inner_model.float().fuse().eval())

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % model_fp)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


class LayoutAnalyzer(object):
    def __init__(
        self,
        model_name: str = 'yolov7_tiny',
        *,
        model_backend: str = 'pytorch',  # ['pytorch', 'onnx']
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        device: str = 'cpu',
        **kwargs,
    ):
        model_backend = model_backend.lower()
        assert model_backend in ('pytorch', 'onnx')
        self._model_name = model_name
        self._model_backend = model_backend

        self.device = select_device(device)

        self._assert_and_prepare_model_files(model_fp, root)
        logger.info('Use model: %s' % self._model_fp)

        self.model = attempt_load(
            self._model_fp, cfg_fp=self._arch_yaml, map_location=self.device
        )  # load FP32 model
        self.model.eval()

        self.categories = CATEGORIES
        self.stride = int(self.model.stride.max())  # model stride
        # self.img_size = check_img_size(image_size, s=self.stride)  # check img_size

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if (self._model_name, self._model_backend) not in LAYOUT_MODELS:
            raise NotImplementedError(
                'model %s is not supported currently'
                % ((self._model_name, self._model_backend),)
            )

        self._arch_yaml = LAYOUT_MODELS[(self._model_name, self._model_backend)][
            'arch_yaml'
        ]
        if model_fp is not None:
            self._model_fp = model_fp
            return

        self._model_dir = os.path.join(root, MODEL_VERSION, LAYOUT_SPACE)
        suffix = 'pt' if self._model_backend == 'pytorch' else 'onnx'
        model_fp = os.path.join(self._model_dir, '%s.%s' % (self._model_name, suffix))
        if not os.path.isfile(model_fp):
            logger.warning('Can NOT find model file %s' % model_fp)
            url = LAYOUT_MODELS[(self._model_name, self._model_backend)]['url']

            get_model_file(url, self._model_dir)  # download the .zip file and unzip

        self._model_fp = model_fp

    def analyze(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        resized_shape: int = 800,
        box_margin: int = 2,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        outs = []
        single = False
        if not isinstance(img_list, list):
            img_list = [img_list]
            single = True

        for img in img_list:
            img, img0 = self._preprocess_images(img, resized_shape)
            outs.append(self.analyze_one(img, img0, box_margin))

        return outs[0] if single else outs

    def _preprocess_images(
        self, img: Union[str, Path, Image.Image, np.ndarray], resized_shape: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            img ():

        Returns: (img, img0)
            * img: RGB-formated ndarray: [3, H, W]
            * img0: BGR-formated ndarray: [H, W, 3]

        """
        if isinstance(img, (str, Path)):
            if not os.path.isfile(img):
                raise FileNotFoundError(img)
            img0 = cv2.imread(img, cv2.IMREAD_COLOR)
        elif isinstance(img, Image.Image):
            img0 = np.asarray(img.convert('RGB'), dtype='float32')
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            img0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError('type %s is not supported now' % str(type(img)))

        img_size = check_img_size(resized_shape, s=self.stride)  # check img_size
        # Padded resize
        img = letterbox(img0, img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0

    @torch.no_grad()
    def analyze_one(
        self, img, img0, box_margin: int = 2,
    ):
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t3 = time_synchronized()

        one_out = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xyxy = self._expand(xyxy, box_margin, img0.shape)
                    one_out.append(
                        {
                            'type': self.categories[int(cls)],
                            'box': xyxy24p(xyxy, np.array),
                            'score': float(conf),
                        }
                    )

            logger.info(
                f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS'
            )

        return one_out

    def _expand(self, xyxy, box_margin, shape):
        xmin, ymin, xmax, ymax = [float(_x) for _x in xyxy]
        xmin = max(0, xmin - box_margin)
        ymin = max(0, ymin - box_margin)
        xmax = min(shape[1], xmax + box_margin)
        ymax = min(shape[0], ymax + box_margin)
        return [xmin, ymin, xmax, ymax]

    def save_img(self, img0, one_out, save_path):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.categories]
        for one_box in one_out:
            _type = one_box['type']
            conf = one_box['score']
            box = one_box['box']
            xyxy = [box[0, 0], box[0, 1], box[2, 0], box[2, 1]]
            label = f'{_type} {conf:.2f}'
            plot_one_box(
                xyxy,
                img0,
                label=label,
                color=colors[self.categories.index(_type)],
                line_thickness=1,
            )

        cv2.imwrite(save_path, img0)
        logger.info(f" The image with the result is saved in: {save_path}")


if __name__ == '__main__':
    import argparse
    from ..utils import set_logger

    logger = set_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-fp', type=str, default='yolov7.pt', help='model.pt path(s)'
    )
    parser.add_argument(
        '--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)'
    )
    parser.add_argument(
        '--source', type=str, default='inference/images', help='source'
    )  # file/folder, 0 for webcam
    parser.add_argument(
        '--img-size', type=int, default=640, help='inference size (pixels)'
    )
    parser.add_argument(
        '--conf-thres', type=float, default=0.25, help='object confidence threshold'
    )
    parser.add_argument(
        '--iou-thres', type=float, default=0.45, help='IOU threshold for NMS'
    )
    parser.add_argument(
        '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument(
        '--save-conf', action='store_true', help='save confidences in --save-txt labels'
    )
    parser.add_argument(
        '--nosave', action='store_true', help='do not save images/videos'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        help='filter by class: --class 0, or --class 0 2 3',
    )
    parser.add_argument(
        '--agnostic-nms', action='store_true', help='class-agnostic NMS'
    )
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument(
        '--project', default='runs/detect', help='save results to project/name'
    )
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='existing project/name ok, do not increment',
    )
    opt = parser.parse_args()
    logger.info(opt)

    analyzer = LayoutAnalyzer(model_fp=opt.model_fp)
    out = analyzer.analyze(opt.source, resized_shape=opt.img_size)
    img0 = cv2.imread(opt.source, cv2.IMREAD_COLOR)
    analyzer.save_img(img0, out, 'out-' + os.path.basename(opt.source))
