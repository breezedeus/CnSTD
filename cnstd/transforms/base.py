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

import random
from typing import List, Any, Callable, Dict, Tuple
import numpy as np

from ..utils import normalize_img_array
from ..utils.repr import NestedObject
from .utils import invert_colors, rotate


__all__ = ['NormalizeAug', 'ColorInversion', 'OneOf', 'RandomApply', 'RandomRotate']


class NormalizeAug(object):
    def __call__(self, img):
        return normalize_img_array(img)


class ColorInversion(NestedObject):
    """Applies the following tranformation to a tensor (image or batch of images):
    convert to grayscale, colorize (shift 0-values randomly), and then invert colors

    Example::
        >>> transfo = ColorInversion(min_val=0.6)
        >>> out = transfo(tf.random.uniform(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        min_val: range [min_val, 1] to colorize RGB pixels
    """
    def __init__(self, min_val: float = 0.5) -> None:
        self.min_val = min_val

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}"

    def __call__(self, img: Any) -> Any:
        return invert_colors(img, self.min_val)


class OneOf(NestedObject):
    """Randomly apply one of the input transformations

    Example::
        >>> transfo = OneOf([JpegQuality(), Gamma()])
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transforms: list of transformations, one only will be picked
    """

    _children_names: List[str] = ['transforms']

    def __init__(self, transforms: List[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, img: Any) -> Any:
        # Pick transformation
        transfo = self.transforms[int(random.random() * len(self.transforms))]
        # Apply
        return transfo(img)


class RandomApply(NestedObject):
    """Apply with a probability p the input transformation

    Example::
        >>> transfo = RandomApply(Gamma(), p=.5)
        >>> out = transfo(tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transform: transformation to apply
        p: probability to apply
    """
    def __init__(self, transform: Callable[[Any], Any], p: float = .5) -> None:
        self.transform = transform
        self.p = p

    def extra_repr(self) -> str:
        return f"transform={self.transform}, p={self.p}"

    def __call__(self, img: Any) -> Any:
        if random.random() < self.p:
            return self.transform(img)
        return img


class RandomRotate(NestedObject):
    """Randomly rotate a tensor image

    Args:
        max_angle: maximum angle for rotation, in degrees. Angles will be uniformly picked in
        [-max_angle, max_angle]
    """
    def __init__(self, max_angle: float = 25.) -> None:
        self.max_angle = max_angle

    def extra_repr(self) -> str:
        return f"max_angle={self.max_angle}"

    def __call__(self, img: Any, target: Dict[str, np.ndarray]) -> Tuple[Any, Dict[str, np.ndarray]]:
        angle = random.uniform(-self.max_angle, self.max_angle)
        img, target['boxes'] = rotate(img, target['boxes'], angle)
        return img, target
