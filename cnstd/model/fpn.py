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
from typing import List

import torch
from torch import nn
from torchvision.ops.deform_conv import DeformConv2d

logger = logging.getLogger(__name__)


class FeaturePyramidNetwork(nn.Module):
    def __init__(
            self, in_channels: List[int], out_channels: int, deform_conv: bool = False,
    ) -> None:

        super().__init__()

        out_chans = out_channels // len(in_channels)

        conv_layer = DeformConv2d if deform_conv else nn.Conv2d

        self.in_branches = nn.ModuleList(
            [
                nn.Sequential(
                    conv_layer(chans, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for idx, chans in enumerate(in_channels)
            ]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_branches = nn.ModuleList(
            [
                nn.Sequential(
                    conv_layer(out_channels, out_chans, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_chans),
                    nn.ReLU(inplace=True),
                    nn.Upsample(
                        scale_factor=2 ** idx, mode='bilinear', align_corners=True
                    ),
                )
                for idx, chans in enumerate(in_channels)
            ]
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) != len(self.out_branches):
            raise AssertionError
        # Conv1x1 to get the same number of channels
        _x: List[torch.Tensor] = [branch(t) for branch, t in zip(self.in_branches, x)]
        out: List[torch.Tensor] = self._merge(_x)

        # Conv and final upsampling
        out = [branch(t) for branch, t in zip(self.out_branches, out[::-1])]

        return torch.cat(out, dim=1)

    def _merge(self, _x: List[torch.Tensor]) -> List[torch.Tensor]:
        return self._merge_small_to_large(_x)

    def _merge_small_to_large(self, _x: List[torch.Tensor]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = [_x[-1]]
        for t in _x[:-1][::-1]:
            out.append(self.upsample(out[-1]) + t)
        return out


class PathAggregationNetwork(FeaturePyramidNetwork):
    """
    参考：https://github.dev/RangiLyu/nanodet 。
    This is an implementation of the `PAN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>` .
    """
    def __init__(
            self, in_channels: List[int], out_channels: int, deform_conv: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, deform_conv)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def _merge(self, _x: List[torch.Tensor]) -> List[torch.Tensor]:
        _x = self._merge_small_to_large(_x)
        return self._merge_large_to_small(_x)

    def _merge_large_to_small(self, _x: List[torch.Tensor]) -> List[torch.Tensor]:
        out = [v for v in _x]
        for i in range(len(out)-1, 1, -1):
            out[i-1] += self.downsample(out[i])
        return out
