# coding: utf-8
import string
from .__version__ import __version__


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '1.2.*'，对应的 MODEL_VERSION 都是 '1.2.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2]) + '.0'

root_url = (
    'https://static.einplus.cn/cnocr/%s'
    % MODEL_VERSION
)
# name: (epochs, url)
AVAILABLE_MODELS = {
    # 'conv-lite-fc': (25, root_url + '/conv-lite-fc.zip'),
}
