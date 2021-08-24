# coding: utf-8
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
from .__version__ import __version__


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '1.2.*'，对应的 MODEL_VERSION 都是 '1.2.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2]) + '.0'
BACKBONE_NET_NAME = ['mobilenetv3', 'resnet50_v1b']

root_url = (
    'https://beiye-model.oss-cn-beijing.aliyuncs.com/models/cnstd/%s/'
    % MODEL_VERSION
)
# name: (epochs, url)
AVAILABLE_MODELS = {
    'mobilenetv3': (59, root_url + '/mobilenetv3.zip'),
    'resnet50_v1b': (49, root_url + '/resnet50_v1b.zip'),
}
