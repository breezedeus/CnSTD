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

from typing import Tuple, List

__all__ = ['Point2D', 'BoundingBox', 'RotatedBbox', 'Polygon4P', 'Polygon']


Point2D = Tuple[float, float]
BoundingBox = Tuple[Point2D, Point2D]
RotatedBbox = Tuple[float, float, float, float, float]
Polygon4P = Tuple[Point2D, Point2D, Point2D, Point2D]
Polygon = List[Point2D]
