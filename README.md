# 最近更新 【2020.06.01】：V0.1.0

初次提交，主要功能：

* 利用PSENet进行场景文字检测（STD），支持两种backbone模型：`mobilenetv3` 和 `resnet50_v1b`。



# cnstd

**cnstd**是**Python 3**下的场景文字检测（STD）工具包，自带了多个训练好的检测模型，安装后即可直接使用。当前的文字检测模型使用的是[PSENet](https://github.com/whai362/PSENet)，目前支持两种backbone模型：`mobilenetv3` 和 `resnet50_v1b`。它们都是在ICPR和ICDAR15训练数据上训练得到的。



本项目初始代码来自 [saicoco/Gluon-PSENet](https://github.com/saicoco/Gluon-PSENet) ，感谢作者。



## 示例







## 安装

嗯，安装真的很简单。

```bash
pip install cnstd
```

> 注意：请使用Python3 (3.4, 3.5, 3.6以及之后版本应该都行)，没测过Python2下是否ok。

(注：依赖opencv，所以可能需要额外安装opencv。)



## 已有模型





## 使用方法

首次使用cnocr时，系统会自动下载zip格式的模型压缩文件，并存于 `~/.cnstd`目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\cnstd`）。
下载后的zip文件代码会自动对其解压，然后把解压后的模型相关目录放于`~/.cnstd/0.1.0`目录中。

如果系统无法自动成功下载zip文件，则需要手动从 **[cnocr-models](https://github.com/breezedeus/cnocr-models)** 下载此zip文件并把它放于 `~/.cnstd/0.1.0`目录。如果Github下载太慢，也可以从 [百度云盘](https://pan.baidu.com/s/1CnbtE693FQqKM1pBDTCrNg) 下载， 提取码为 ` msua`。

放置好zip文件后，后面的事代码就会自动执行了。



### 图片预测

类`CnOcr`是OCR的主类，包含了三个函数针对不同场景进行文字识别。类`CnOcr`的初始化函数如下：

```python
class CnOcr(object):
    def __init__(
        self,
        model_name='densenet-lite-fc',
        model_epoch=None,
        cand_alphabet=None,
        root=data_dir(),
        context='cpu',
        name=None,
    ):
```

其中的几个参数含义如下：

* `model_name`: 模型名称，即上面表格第一列中的值。默认为 `densenet-lite-fc`。
* `model_epoch`: 模型迭代次数。默认为 `None`，表示使用默认的迭代次数值。对于模型名称 `densenet-lite-fc`就是 `40`。
* `root`: 模型文件所在的根目录。
  * Linux/Mac下默认值为 `~/.cnocr`，表示模型文件所处文件夹类似 `~/.cnocr/1.2.0/densenet-lite-fc`。
  * Windows下默认值为 `C:\Users\<username>\AppData\Roaming\cnocr`。
* `context`：预测使用的机器资源，可取值为字符串`cpu`、`gpu`，或者 `mx.Context`实例。
* `name`：正在初始化的这个实例的名称。如果需要同时初始化多个实例，需要为不同的实例指定不同的名称。



每个参数都有默认取值，所以可以不传入任何参数值进行初始化：`ocr = CnOcr()`。




类`CnOcr`主要包含三个函数，下面分别说明。



#### 1. 函数`CnOcr.ocr(img_fp)`

函数`CnOcr.ocr(img_fp)`可以对包含多行文字（或单行）的图片进行文字识别。



**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的图片文件路径（如上例）；或者是已经从图片文件中读入的数组，类型可以为`mx.nd.NDArray` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是`(height, width, 3)`，第三个维度是channel，它应该是`RGB`格式的。
- 返回值：为一个嵌套的`list`，类似这样`[['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]`。



**调用示例**：


```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr('examples/multi-line_cn1.png')
print("Predicted Chars:", res)
```

或：

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = mx.image.imread(img_fp, 1)
res = ocr.ocr(img)
print("Predicted Chars:", res)
```



上面使用的图片文件 [examples/multi-line_cn1.png](./examples/multi-line_cn1.png)内容如下：

![examples/multi-line_cn1.png](/Users/king/Documents/WhatIHaveDone/Test/cnocr/examples/multi-line_cn1.png)



上面预测代码段的返回结果如下：

```bash
Predicted Chars: [['网', '络', '支', '付', '并', '无', '本', '质', '的', '区', '别', '，', '因', '为'],
                  ['每', '一', '个', '手', '机', '号', '码', '和', '邮', '件', '地', '址', '背', '后'],
                  ['都', '会', '对', '应', '着', '一', '个', '账', '户', '一', '―', '这', '个', '账'],
                  ['户', '可', '以', '是', '信', '用', '卡', '账', '户', '、', '借', '记', '卡', '账'],
                  ['户', '，', '也', '包', '括', '邮', '局', '汇', '款', '、', '手', '机', '代'],
                  ['收', '、', '电', '话', '代', '收', '、', '预', '付', '费', '卡', '和', '点', '卡'],
                  ['等', '多', '种', '形', '式', '。']]
```



#### 2. 函数`CnOcr.ocr_for_single_line(img_fp)`

如果明确知道要预测的图片中只包含了单行文字，可以使用函数`CnOcr.ocr_for_single_line(img_fp)`进行识别。和 `CnOcr.ocr()`相比，`CnOcr.ocr_for_single_line()`结果可靠性更强，因为它不需要做额外的分行处理。

**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的单行文字图片文件路径（如上例）；或者是已经从图片文件中读入的数组，类型可以为`mx.nd.NDArray` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是`(height, width)`或`(height, width, channel)`。如果没有channel，表示传入的就是灰度图片。第三个维度channel可以是`1`（灰度图片）或者`3`（彩色图片）。如果是彩色图片，它应该是`RGB`格式的。
- 返回值：为一个`list`，类似这样`['你', '好']`。



**调用示例**：

```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr_for_single_line('examples/rand_cn1.png')
print("Predicted Chars:", res)
```

或：

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/rand_cn1.png'
img = mx.image.imread(img_fp, 1)
res = ocr.ocr_for_single_line(img)
print("Predicted Chars:", res)
```


对图片文件 [examples/rand_cn1.png](./examples/rand_cn1.png)：

![examples/rand_cn1.png](/Users/king/Documents/WhatIHaveDone/Test/cnocr/examples/rand_cn1.png)

的预测结果如下：

```bash
Predicted Chars: ['笠', '淡', '嘿', '骅', '谧', '鼎', '皋', '姚', '歼', '蠢', '驼', '耳', '胬', '挝', '涯', '狗', '蒽', '子', '犷'] 
```



#### 3. 函数`CnOcr.ocr_for_single_lines(img_list)`

函数`CnOcr.ocr_for_single_lines(img_list)`可以**对多个单行文字图片进行批量预测**。函数`CnOcr.ocr(img_fp)`和`CnOcr.ocr_for_single_line(img_fp)`内部其实都是调用的函数`CnOcr.ocr_for_single_lines(img_list)`。



**函数说明**：

- 输入参数` img_list`: 为一个`list`；其中每个元素是已经从图片文件中读入的数组，类型可以为`mx.nd.NDArray` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是`(height, width)`或`(height, width, channel)`。如果没有channel，表示传入的就是灰度图片。第三个维度channel可以是`1`（灰度图片）或者`3`（彩色图片）。如果是彩色图片，它应该是`RGB`格式的。
- 返回值：为一个嵌套的`list`，类似这样`[['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]`。



**调用示例**：

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = mx.image.imread(img_fp, 1).asnumpy()
line_imgs = line_split(img, blank=True)
line_img_list = [line_img for line_img, _ in line_imgs]
res = ocr.ocr_for_single_lines(line_img_list)
print("Predicted Chars:", res)
```



更详细的使用方法，可参考 [tests/test_cnocr.py](./tests/test_cnocr.py) 中提供的测试用例。



### 脚本引用

也可以使用脚本模式预测：

```bash
python scripts/cnocr_predict.py --file examples/multi-line_cn1.png
```

返回结果同上面。



### 训练自己的模型

cnocr自带训练好的模型， 安装后即可直接使用。但如果你需要训练自己的模型，请参考下面的步骤。所有代码均可在文件 [Makefile](./Makefile) 中找到。



#### （一）转换图片数据格式

为了提升训练效率，在开始训练之前，需要使用mxnet的`recordio`首先把数据转换成二进制格式：

```makefile
DATA_ROOT_DIR = data/sample-data
REC_DATA_ROOT_DIR = data/sample-data-lst

# `EMB_MODEL_TYPE` 可取值：['conv', 'conv-lite-rnn', 'densenet', 'densenet-lite']
EMB_MODEL_TYPE = densenet-lite
# `SEQ_MODEL_TYPE` 可取值：['lstm', 'gru', 'fc']
SEQ_MODEL_TYPE = fc
MODEL_NAME = $(EMB_MODEL_TYPE)-$(SEQ_MODEL_TYPE)

# 产生 *.lst 文件
gen-lst:
    python scripts/im2rec.py --list --num-label 20 --chunks 1 \
        --train-idx-fp $(DATA_ROOT_DIR)/train.txt --test-idx-fp $(DATA_ROOT_DIR)/test.txt --prefix $(REC_DATA_ROOT_DIR)/sample-data

# 利用 *.lst 文件产生 *.idx 和 *.rec 文件。
# 真正的图片文件存储在 `examples` 目录，可通过 `--root` 指定。
gen-rec:
    python scripts/im2rec.py --pack-label --color 1 --num-thread 1 --prefix $(REC_DATA_ROOT_DIR) --root examples
```



#### （二）训练模型

利用下面命令在CPU上训练模型：

```makefile
# 训练模型
train:
    python scripts/cnocr_train.py --gpu 0 --emb_model_type $(EMB_MODEL_TYPE) --seq_model_type $(SEQ_MODEL_TYPE) \
        --optimizer adam --epoch 20 --lr 1e-4 \
        --train_file $(REC_DATA_ROOT_DIR)/sample-data_train --test_file $(REC_DATA_ROOT_DIR)/sample-data_test
```

如果需要在GPU上训练，把上面命令中的参数 `--gpu 0`改为`--gpu <num_gpu>`，其中的`<num_gpu>` 为使用的GPU数量。注意，使用GPU训练需要安装mxnet的GPU版本，如`mxnet-cu101`。



#### （三）评估模型

评估模型的代码依赖一些额外的python包，使用下面命令安装这些额外的包：

```bash
pip install cnocr[dev]
```



训练好的模型，可以使用脚本 [scripts/cnocr_evaluate.py](scripts/cnocr_evaluate.py) 评估在测试集上的效果：

```makefile
# 在测试集上评估模型，所有badcases的具体信息会存放到文件夹 `evaluate/$(MODEL_NAME)` 中
evaluate:
    python scripts/cnocr_evaluate.py --model-name $(MODEL_NAME) --model-epoch 1 -v -i $(DATA_ROOT_DIR)/test.txt \
        --image-prefix-dir examples --batch-size 128 -o evaluate/$(MODEL_NAME)
```



当然，也可以查看模型在单个文件上的预测效果：

```makefile
predict:
    python scripts/cnocr_predict.py --model_name $(MODEL_NAME) --file examples/rand_cn1.png
```



上面所有代码均可在文件 [Makefile](./Makefile) 中找到。



## 未来工作

* [x] 支持图片包含多行文字 (`Done`)
* [x] crnn模型支持可变长预测，提升灵活性 (since `V1.0.0`)
* [x] 完善测试用例 (`Doing`)
* [x] 修bugs（目前代码还比较凌乱。。） (`Doing`)
* [x] 支持`空格`识别（since `V1.1.0`）
* [x] 尝试新模型，如 DenseNet，进一步提升识别准确率（since `V1.1.0`）

