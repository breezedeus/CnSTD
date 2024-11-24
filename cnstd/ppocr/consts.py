# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
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


MODEL_LABELS_FILE_DICT = {
    ('ch_PP-OCRv3_det', 'onnx'): {
        'url': 'ch_PP-OCRv3_det_infer-onnx.zip',
    },
    ('ch_PP-OCRv2_det', 'onnx'): {
        'url': 'ch_PP-OCRv2_det_infer-onnx.zip',
    },
    ('en_PP-OCRv3_det', 'onnx'): {
        'url': 'en_PP-OCRv3_det_infer-onnx.zip',
        'detector': 'RapidDetector',
        'repo': 'breezedeus/cnstd-ppocr-en_PP-OCRv3_det',
    },
    ('ch_PP-OCRv4_det', 'onnx'): {
        'detector': 'RapidDetector',
        'repo': 'breezedeus/cnstd-ppocr-ch_PP-OCRv4_det',
    },
    ('ch_PP-OCRv4_det_server', 'onnx'): {
        'detector': 'RapidDetector',
        'repo': 'breezedeus/cnstd-ppocr-ch_PP-OCRv4_det_server',
    },
}

PP_SPACE = 'ppocr'
