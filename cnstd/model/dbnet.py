# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
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
# Credits: adapted from https://github.com/mindee/doctr

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.deform_conv import DeformConv2d

from .base import DBPostProcessor, _DBNet
from .fpn import FeaturePyramidNetwork, PathAggregationNetwork

__all__ = ['DBNet', 'gen_dbnet']


logger = logging.getLogger(__name__)


class DBNet(_DBNet, nn.Module):
    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        fpn_channels: List[int],
        *,
        head_chans: int = 256,
        deform_conv: bool = False,
        fpn_type: str = 'fpn',
        num_classes: int = 1,
        auto_rotate_whole_image=False,
        rotated_bbox: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:

        super().__init__()
        self.cfg = cfg

        if len(feat_extractor.return_layers) != len(fpn_channels):
            raise AssertionError

        conv_layer = DeformConv2d if deform_conv else nn.Conv2d

        self.rotated_bbox = rotated_bbox

        self.feat_extractor = feat_extractor
        fpn_cls = FeaturePyramidNetwork if fpn_type == 'fpn' else PathAggregationNetwork
        self.fpn = fpn_cls(fpn_channels, head_chans, deform_conv)
        # Conv1 map to channels

        self.prob_head = nn.Sequential(
            conv_layer(head_chans, head_chans // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                head_chans // 4, head_chans // 4, 2, stride=2, bias=False
            ),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, 1, 2, stride=2),
        )
        self.thresh_head = nn.Sequential(
            conv_layer(head_chans, head_chans // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                head_chans // 4, head_chans // 4, 2, stride=2, bias=False
            ),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, num_classes, 2, stride=2),
        )

        self.postprocessor = DBPostProcessor(
            auto_rotate_whole_image=auto_rotate_whole_image,
            rotated_bbox=self.rotated_bbox,
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, DeformConv2d)):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def calculate_loss(self, batch, **kwargs):
        return self(
            batch['image'],
            batch['gt'],
            batch['mask'],
            batch['thresh_map'],
            batch['thresh_mask'],
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,  # [N, C, H, W]
        seg_target: Optional[torch.Tensor] = None,  # [N, H, W]
        seg_mask: Optional[torch.Tensor] = None,  # [N, H, W]
        thresh_target: Optional[torch.Tensor] = None,  # [N, H, W]
        thresh_mask: Optional[torch.Tensor] = None,  # [N, H, W]
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """

        Args:
            x:
            seg_target:
            seg_mask:
            thresh_target:
            thresh_mask:
            return_model_output:
            return_preds:

        Returns: dict;
            "out_map": prob tensor
            "preds": list; [List[boxes tensor], List[angles tensor]]
                boxes tensor: 5 (rotated_bbox==False) or 6 (rotated_bbox==True) columns;
                    * containing [xmin, ymin, xmax, ymax, score] for the box (rotated_bbox==False);
                    * containing [x, y, w, h, angle, score] for the box (rotated_bbox==True)
                angles tensor: N angles (page orientations, each for one page or image).
            "loss": scalar tensor

        """
        # Extract feature maps at different stages
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        # Pass through the FPN
        feat_concat = self.fpn(feats)
        logits = self.prob_head(feat_concat)

        out: Dict[str, Any] = {}
        if return_model_output or seg_target is None or return_preds:
            prob_map = torch.sigmoid(logits)

        if return_model_output:
            out["out_map"] = prob_map

        if seg_target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(
                prob_map.squeeze(1).detach().cpu().numpy().astype(np.float32)
            )

        if seg_target is not None:
            thresh_map = self.thresh_head(feat_concat)
            loss = self.compute_loss(
                logits, thresh_map, seg_target, seg_mask, thresh_target, thresh_mask
            )
            out['loss'] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        thresh_map: torch.Tensor,
        seg_target: Optional[torch.Tensor],  # [N, H, W]
        seg_mask: Optional[torch.Tensor],  # [N, H, W]
        thresh_target: Optional[torch.Tensor],  # [N, H, W]
        thresh_mask: Optional[torch.Tensor],  # [N, H, W]
    ) -> torch.Tensor:
        """Compute a batch of gts, masks, thresh_gts, thresh_masks from a list of boxes
        and a list of masks for each image. From there it computes the loss with the model output

        Args:
            out_map: output feature map of the model of shape (N, C, H, W)
            thresh_map: threshold map of shape (N, C, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry

        Returns:
            A loss tensor
        """

        prob_map = torch.sigmoid(out_map.squeeze(1))
        thresh_map = torch.sigmoid(thresh_map.squeeze(1))

        # targets = self.compute_target(target, prob_map.shape)  # type: ignore[arg-type]

        # seg_target, seg_mask = torch.from_numpy(targets[0]), torch.from_numpy(targets[1])
        # seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)
        # thresh_target, thresh_mask = torch.from_numpy(targets[2]), torch.from_numpy(targets[3])
        # thresh_target, thresh_mask = thresh_target.to(out_map.device), thresh_mask.to(out_map.device)

        # Compute balanced BCE loss for proba_map
        bce_scale = 5.0
        bce_loss = F.binary_cross_entropy_with_logits(
            out_map.squeeze(1), seg_target, reduction='none'
        )[seg_mask]

        neg_target = 1 - seg_target[seg_mask]
        positive_count = seg_target[seg_mask].sum()
        negative_count = torch.minimum(neg_target.sum(), 3.0 * positive_count)
        negative_loss = bce_loss * neg_target
        negative_loss = negative_loss.sort().values[-int(negative_count.item()) :]
        sum_losses = torch.sum(bce_loss * seg_target[seg_mask]) + torch.sum(
            negative_loss
        )
        balanced_bce_loss = sum_losses / (positive_count + negative_count + 1e-6)

        # Compute dice loss for approxbin_map
        bin_map = 1 / (
            1 + torch.exp(-50.0 * (prob_map[seg_mask] - thresh_map[seg_mask]))
        )

        bce_min = bce_loss.min()
        weights = (bce_loss - bce_min) / (bce_loss.max() - bce_min) + 1.0
        inter = torch.sum(bin_map * seg_target[seg_mask] * weights)
        union = torch.sum(bin_map) + torch.sum(seg_target[seg_mask]) + 1e-8
        dice_loss = 1 - 2.0 * inter / union

        # Compute l1 loss for thresh_map
        l1_scale = 10.0
        if torch.any(thresh_mask):
            l1_loss = torch.mean(
                torch.abs(thresh_map[thresh_mask] - thresh_target[thresh_mask])
            )
        else:
            l1_loss = torch.zeros(1)

        return l1_scale * l1_loss + bce_scale * balanced_bce_loss + dice_loss


def gen_dbnet(
    config: Dict[str, Any],
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    **kwargs: Any,
) -> DBNet:

    pretrained_backbone = pretrained_backbone and not pretrained
    logger.info('config for "gen_dbnet": %s' % config)

    # Feature extractor
    backbone = config['backbone'](pretrained=pretrained_backbone)
    if isinstance(config['backbone_submodule'], str):
        backbone = getattr(backbone, config['backbone_submodule'])
    feat_extractor = IntermediateLayerGetter(
        backbone,
        {layer_name: str(idx) for idx, layer_name in enumerate(config['fpn_layers'])},
    )

    # Build the model
    model = DBNet(feat_extractor, config['fpn_channels'], cfg=config, **kwargs)
    # Load pretrained parameters
    if pretrained:
        raise NotImplementedError

    return model
