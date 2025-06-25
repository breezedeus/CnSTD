#!/usr/bin/env python3
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
from setuptools import find_packages, setup
from pathlib import Path

PACKAGE_NAME = "cnstd"

here = Path(__file__).parent

long_description = (here / "README.md").read_text(encoding="utf-8")

about = {}
exec(
    (here / PACKAGE_NAME.replace('.', os.path.sep) / "__version__.py").read_text(
        encoding="utf-8"
    ),
    about,
)

required = [
    'click',
    'tqdm',
    'pyyaml',
    'unidecode',
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    'numpy',
    'scipy',
    'pandas',
    "pytorch-lightning",
    'pillow>=5.3.0',
    'opencv-python>=4.0.0',
    'shapely',
    # 'Polygon3',
    'pyclipper',
    'matplotlib',
    'seaborn',
    "onnx",
    "huggingface_hub",
    "ultralytics",
    "rapidocr>=3.0",
]

extras_require = {
    "ort-cpu": ["onnxruntime"],
    "ort-gpu": ["onnxruntime-gpu"],
    "dev": ["pip-tools", "pytest"],
}

entry_points = """
[console_scripts]
cnstd = cnstd.cli:cli
"""

setup(
    name=PACKAGE_NAME,
    version=about['__version__'],
    description="Python3 package for Chinese/English Scene Text Detection (STD), Mathematical Formula Detection (MFD), "
                "and Layout Analysis, with free pretrained models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='breezedeus',
    author_email='breezedeus@163.com',
    license='Apache 2.0',
    url='https://github.com/breezedeus/cnstd',
    platforms=["Mac", "Linux", "Windows"],
    packages=find_packages(),
    entry_points=entry_points,
    include_package_data=True,
    data_files=[
        (
            '',
            [
                'cnstd/yolov7/yolov7-tiny-layout.yaml',
                'cnstd/yolov7/yolov7-tiny-mfd.yaml',
                'cnstd/yolov7/yolov7-mfd.yaml',
            ],
        )
    ],
    install_requires=required,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
