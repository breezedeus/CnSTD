# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from typing import List, Any, Optional, Dict, Tuple

import cv2
from PIL import Image
import torch
import torchvision.transforms as T

from ..utils import pil_to_numpy, normalize_img_array, restore_img, imsave
from ..utils.repr import NestedObject
from .._utils import rotate_page, get_bitmap_angle, extract_crops, extract_rcrops
from ..preprocessor import PreProcessor


__all__ = ['DetectionModel', 'DetectionPostProcessor', 'DetectionPredictor']


class DetectionModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        min_size_box (int): minimal length (pix) to keep a box
        max_candidates (int): maximum boxes to consider in a single page
        box_thresh (float): minimal objectness score to consider a box
    """

    def __init__(
        self,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.5,
        rotated_bbox: bool = False,
    ) -> None:

        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.rotated_bbox = rotated_bbox

    def extra_repr(self) -> str:
        return f"box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(
        pred: np.ndarray, points: np.ndarray, rotated_bbox: bool = False
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]

        if not rotated_bbox:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin : ymax + 1, xmin : xmax + 1].mean()

        else:
            mask = np.zeros((h, w), np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def bitmap_to_boxes(self, pred: np.ndarray, bitmap: np.ndarray,) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, proba_map: np.ndarray,) -> Tuple[List[np.ndarray], List[float]]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, H, W)

        returns:
            list of N tensors (for each input sample), with each tensor of shape (*, 5) or (*, 6),
            and a list of N angles (page orientations).
        """

        bitmap = (proba_map > self.bin_thresh).astype(proba_map.dtype)

        boxes_batch, angles_batch = [], []
        # Kernel for opening, empirical law for ksize
        k_size = 1 + int(proba_map[0].shape[0] / 512)
        kernel = np.ones((k_size, k_size), np.uint8)

        for p_, bitmap_ in zip(proba_map, bitmap):
            # Perform opening (erosion + dilatation)
            bitmap_ = cv2.morphologyEx(bitmap_, cv2.MORPH_OPEN, kernel)
            # Rotate bitmap and proba_map
            angle = get_bitmap_angle(bitmap_)
            angles_batch.append(angle)
            bitmap_, p_ = rotate_page(bitmap_, -angle), rotate_page(p_, -angle)
            boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
            boxes_batch.append(boxes)

        return boxes_batch, angles_batch


class DetectionPredictor(NestedObject):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    _children_names: List[str] = ['model']

    def __init__(self, model, debug=False,) -> None:

        self.debug = debug
        self.model = model
        self.model.eval()
        self.val_transform = T.Resize(self.model.cfg['input_shape'][1:])
        self.extract_crops_fn = (
            extract_rcrops if self.model.rotated_bbox else extract_crops
        )

    @torch.no_grad()
    def __call__(
        self, img_list: List[Image.Image], **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[float]]:
        batch = self.preprocess(img_list)

        out = self.model(batch, return_preds=True, **kwargs)  # type:ignore[operator]
        boxes, angles = out['preds']
        # FIXME resize back for boxes and images
        crops_list = []
        scores_list = []
        idx = 0
        for image, _boxes, angle in zip(batch, boxes, angles):
            image = restore_img(image.numpy().transpose((1, 2, 0)))  # res: [H, W, 3]
            rotated_img = np.ascontiguousarray(rotate_page(image, -angle))
            crops = []
            scores_list.append(_boxes[:, -1].tolist())
            for crop in self.extract_crops_fn(rotated_img, _boxes[:, :-1]):
                crops.append(crop)
            crops_list.append(crops)

            if self.debug:
                if _boxes.dtype != np.int:
                    _boxes[:, [0, 2]] *= rotated_img.shape[1]
                    _boxes[:, [1, 3]] *= rotated_img.shape[0]
                for box in _boxes:
                    x, y, w, h, alpha, score = box.astype('float32')
                    if score < 0.5:
                        continue
                    box = cv2.boxPoints(((x, y), (w, h), alpha))
                    box = np.int0(box)
                    cv2.drawContours(rotated_img, [box], 0, (255, 0, 0), 1)
                imsave(rotated_img, 'result-%d.png' % idx, normalized=False)
            idx += 1

        return crops_list, scores_list

    def preprocess(self, pil_img_list: List[Image.Image]) -> torch.Tensor:
        img_list = []
        for pil_img in pil_img_list:
            pil_img = self.val_transform(pil_img)
            img = pil_to_numpy(pil_img)
            img = normalize_img_array(img)
            img_list.append(torch.from_numpy(img))
        return torch.stack(img_list, dim=0)


class OCRPredictor(NestedObject):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    _children_names: List[str] = ['det_predictor', 'reco_predictor', 'doc_builder']

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        # reco_predictor: RecognitionPredictor,
        rotated_bbox: bool = False,
    ) -> None:

        self.det_predictor = det_predictor
        # self.reco_predictor = reco_predictor
        # self.doc_builder = DocumentBuilder(rotated_bbox=rotated_bbox)
        self.extract_crops_fn = extract_rcrops if rotated_bbox else extract_crops

    def __call__(
        self, pages: List[np.ndarray], **kwargs: Any,
    ):

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError(
                "incorrect input shape: all pages are expected to be multi-channel 2D images."
            )

        # Localize text elements
        boxes = self.det_predictor(pages, **kwargs)
        # Crop images, rotate page if necessary
        crops = [
            crop
            for page, (_boxes, angle) in zip(pages, boxes)
            for crop in self.extract_crops_fn(rotate_page(page, -angle), _boxes[:, :-1])
        ]  # type: ignore[operator]
        return crops
        # # Identify character sequences
        # word_preds = self.reco_predictor(crops, **kwargs)
        #
        # # Rotate back boxes if necessary
        # boxes, angles = zip(*boxes)
        # boxes = [rotate_boxes(boxes_page, angle) for boxes_page, angle in zip(boxes, angles)]
        # out = self.doc_builder(boxes, word_preds, [page.shape[:2] for page in pages])
        # return out
