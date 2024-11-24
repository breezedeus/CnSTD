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

from collections import OrderedDict

import numpy as np
from PIL import Image
import streamlit as st

from cnstd import CnStd
from cnstd.utils import plot_for_debugging, pil_to_numpy
from cnstd.consts import AVAILABLE_MODELS as STD_MODELS

try:
    from cnocr import CnOcr
    from cnocr.consts import AVAILABLE_MODELS

    cnocr_available = True
except Exception:
    cnocr_available = False


@st.cache(allow_output_mutation=True)
def get_ocr_model(ocr_model_name):
    if not cnocr_available:
        return None
    model_name, model_backend = ocr_model_name
    return CnOcr(model_name, model_backend=model_backend)


@st.cache(allow_output_mutation=True)
def get_std_model(std_model_name, rotated_bbox, use_angle_clf):
    model_name, model_backend = std_model_name
    return CnStd(
        model_name,
        model_backend=model_backend,
        rotated_bbox=rotated_bbox,
        use_angle_clf=use_angle_clf,
    )


def visualize_std(img, std_out, box_score_thresh):
    img = pil_to_numpy(img).transpose((1, 2, 0)).astype(np.uint8)

    plot_for_debugging(
        img, std_out['detected_texts'], box_score_thresh, './streamlit-app'
    )
    st.subheader('STD Result')
    st.image('./streamlit-app-result.png')
    # st.image('./streamlit-app-crops.png')


def visualize_ocr(ocr, std_out):
    st.empty()
    st.subheader('OCR Result')
    ocr_res = OrderedDict({'文本': []})
    ocr_res['概率值'] = []
    for box_info in std_out['detected_texts']:
        cropped_img = box_info['cropped_img']  # 检测出的文本框
        try:
            ocr_out = ocr.ocr_for_single_line(cropped_img)
            prob, text = ocr_out[1], ocr_out[0]
        except:
            prob, text = 0.0, ''
        ocr_res['概率值'].append(prob)
        ocr_res['文本'].append(text)
    st.table(ocr_res)


def main():
    st.sidebar.header('CnStd 设置')
    models = list(STD_MODELS.all_models())
    models.sort()
    std_model_name = st.sidebar.selectbox(
        '模型名称', models, index=models.index(('ch_PP-OCRv4_det', 'onnx'))
    )
    rotated_bbox = st.sidebar.checkbox('是否检测带角度文本框', value=True)
    use_angle_clf = st.sidebar.checkbox('是否使用角度预测模型校正文本框', value=False)
    st.sidebar.subheader('resize 后图片（长边）大小')
    new_size = st.sidebar.slider('高宽尺寸', min_value=124, max_value=4096, value=768)
    st.sidebar.subheader('检测参数')
    box_score_thresh = st.sidebar.slider(
        '得分阈值（低于阈值的结果会被过滤掉）', min_value=0.05, max_value=0.95, value=0.3
    )
    min_box_size = st.sidebar.slider(
        '框大小阈值（更小的文本框会被过滤掉）', min_value=4, max_value=50, value=10
    )
    std = get_std_model(std_model_name, rotated_bbox, use_angle_clf)

    if cnocr_available:
        st.sidebar.markdown("""---""")
        st.sidebar.header('CnOcr 设置')
        all_models = list(AVAILABLE_MODELS.all_models())
        all_models.sort()
        idx = all_models.index(('densenet_lite_136-fc', 'onnx'))
        ocr_model_name = st.sidebar.selectbox('选择模型', all_models, index=idx)
        ocr = get_ocr_model(ocr_model_name)

    st.markdown(
        '# 开源文本检测和识别工具 [CnStd](https://github.com/breezedeus/cnstd) 和 '
        '[CnOcr](https://github.com/breezedeus/cnocr) 演示 Demo'
    )
    st.subheader('选择待检测图片')
    content_file = st.file_uploader('', type=["png", "jpg", "jpeg", "webp"])
    if content_file is None:
        st.stop()

    try:
        img = Image.open(content_file)

        std_out = std.detect(
            img,
            resized_shape=new_size,
            preserve_aspect_ratio=True,
            box_score_thresh=box_score_thresh,
            min_box_size=min_box_size,
        )
        visualize_std(img, std_out, box_score_thresh)

        if cnocr_available:
            visualize_ocr(ocr, std_out)
    except Exception as e:
        st.error(e)


if __name__ == '__main__':
    main()
