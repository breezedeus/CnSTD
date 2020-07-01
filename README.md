# cnstd

**cnstd** 是 **Python 3** 下的**场景文字检测**（**Scene Text Detection**，简称**STD**）工具包，自带了多个训练好的检测模型，安装后即可直接使用。当前的文字检测模型使用的是 [PSENet](https://github.com/whai362/PSENet)，目前支持两种 backbone 模型：**`mobilenetv3`** 和 **`resnet50_v1b`**。它们都是在 **ICPR** 和 **ICDAR15** 的 `11000` 张训练集图片上训练得到的。



如需要识别文本框中的文字，可以结合 **OCR** 工具包 **[cnocr](https://github.com/breezedeus/cnocr)** 一起使用。



本项目初始代码主要来自 [saicoco/Gluon-PSENet](https://github.com/saicoco/Gluon-PSENet) ，感谢作者。



## 示例

![文本检测示例](./docs/cases.png)





## 安装

嗯，安装真的很简单。

```bash
pip install cnstd
```

【注意】：

* 请使用Python3 (3.4, 3.5, 3.6以及之后版本应该都行)，没测过Python2下是否ok。
* 依赖opencv，所以可能需要额外安装opencv。



## 已有模型

当前的文字检测模型使用的是[PSENet](https://github.com/whai362/PSENet)，目前包含两个已训练好的模型，分别对应两种backbone模型：`mobilenetv3` 和 `resnet50_v1b`。它们都是在ICPR和ICDAR15训练数据上训练得到的。


| 模型名称     | backbone模型 | 模型大小 | 迭代次数 |
| ------------ | ------------ | -------- | -------- |
| resnet50_v1b | resnet50_v1b | 121M     | 49       |
| mobilenetv3  | mobilenetv3  | 31M      | 59       |

模型  `resnet50_v1b` 精度略高于模型 `mobilenetv3`。



## 使用方法

首次使用 **cnstd** 时，系统会自动下载zip格式的模型压缩文件，并存放于 `~/.cnstd`目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\cnstd`）。
下载后的zip文件代码会自动对其解压，然后把解压后的模型相关目录放于`~/.cnstd/0.1.0`目录中。

如果系统无法自动成功下载zip文件，则需要手动从 [百度云盘](https://pan.baidu.com/s/1baAbek7gJ8ScYctB-oCu9w)（提取码为 ` 4ndj`）下载对应的zip文件并把它存放于 `~/.cnstd/0.1.0`（Windows下为 `C:\Users\<username>\AppData\Roaming\cnstd\0.1.0`）目录中。放置好zip文件后，后面的事代码就会自动执行了。



### 图片预测

使用类 `CnStd` 进行场景文字的检测。类 `CnStd` 的初始化函数如下：

```python
class CnStd(object):
    """
    场景文字检测器（Scene Text Detection）。虽然名字中有个"Cn"（Chinese），但其实也可以轻松识别英文的。
    """

    def __init__(
        model_name='mobilenetv3',
        model_epoch=None,
        root=data_dir(),
        context='cpu',
        name=None,
    ):
```

其中的几个参数含义如下：

* `model_name`: 模型名称，即上面表格第一列中的值，目前仅支持取值为 `mobilenetv3` 和 `resnet50_v1b`。默认为 `mobilenetv3` 。
* `model_epoch`: 模型迭代次数。默认为 `None`，表示使用系统自带的模型对应的迭代次数。对于模型名称 `mobilenetv3`就是 `59`。
* `root`: 模型文件所在的根目录。
  * Linux/Mac下默认值为 `~/.cnstd`，表示模型文件所处文件夹类似 `~/.cnstd/0.1.0/mobilenetv3`。
  * Windows下默认值为 `C:\Users\<username>\AppData\Roaming\cnstd`。
* `context`：预测使用的机器资源，可取值为字符串`cpu`、`gpu`，或者 `mx.Context`实例。
* `name`：正在初始化的这个实例的名称。如果需要同时初始化多个实例，需要为不同的实例指定不同的名称。



每个参数都有默认取值，所以可以不传入任何参数值进行初始化：`std = CnStd()`。




文本检测使用类`CnOcr`的函数 **`detect()`**，以下是详细说明：



#### 类函数`CnStd.detect(img_fp, max_size, pse_threshold, pse_min_area, **kwargs)`



**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的图片文件路径，或者是已经从图片文件中读入的数组，类型可以为`mx.nd.NDArray` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是`(height, width, 3)`，第三个维度是channel，它应该是`RGB`格式的。

- 输入参数 `max_size`: 如果图片的长边超过这个值，就把图片等比例压缩到长边等于这个size。

- 输入参数 `pse_threshold`: pse中的阈值；越低会导致识别出的文本框越大；反之越小。

- 输入参数 `pse_min_area`: 面积大小低于此值的框会被去掉。所以此值越小，识别出的框可能越多。

- 输入参数 `kwargs`: 目前会使用到的keys有：
              "height_border"，裁切图片时在高度上留出的边界比例，最终上下总共留出的边界大小为height * height_border; 默认为0.05；
              "width_border"，裁切图片时在宽度上留出的边界比例，最终左右总共留出的边界大小为height * width_border; 默认为0.0；
- 返回值：类型为`list`，其中每个元素是一个字典，  存储了检测出的一个框的各种信息。字典包括以下几个值：
	- "box"：检测出的文字对应的矩形框四个点的坐标（第一列对应宽度方向，第二列对应高度方向）；
			`np.ndarray`类型，`shape==(4, 2)`；
		
	- "score"：得分；float类型；分数越高表示越可靠；
	
	- "croppped_img"：对应 "box" 中的图片patch（`RGB`格式），会把倾斜的图片旋转为水平。
		  `np.ndarray`类型，`shape==(width, height, 3)`；
		
	- 示例:
		
```python
		  [{'box': array([[416,  77],
		                  [486,  13],
		                  [800, 325],
		                  [730, 390]], dtype=int32),
		    'score': 1.0, 
		    'cropped_img': array([[[25, 20, 24],
		                           [26, 21, 25],
		                           [25, 20, 24],
		                          ...,
		                           [11, 11, 13],
		                           [11, 11, 13],
		                           [11, 11, 13]]], dtype=uint8)},
		   ...
		  ]
```



#### 调用示例


```python
from cnstd import CnStd
std = CnStd()
box_info_list = std.detect('examples/taobao.jpg')
```

或：

```python
import mxnet as mx
from cnstd import CnStd

std = CnStd()
img_fp = 'examples/taobao.jpg'
img = mx.image.imread(img_fp, 1)
box_info_list = std.detect(img)
```



### 识别检测框中的文字（OCR）

上面示例识别结果中"cropped_img"对应的值可以直接交由 **[cnocr](https://github.com/breezedeus/cnocr)** 中的 **`CnOcr`** 进行文字识别。如上例可以结合  **`CnOcr`** 进行文字识别：

```python
from cnstd import CnStd
from cnocr import CnOcr

std = CnStd()
cn_ocr = CnOcr()

box_info_list = std.detect('examples/taobao.jpg')

for box_info in box_info_list:
    cropped_img = box_info['cropped_img']
    ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
    print('ocr result: %s' % ''.join(ocr_res))
```

注：运行上面示例需要先安装 **`cnocr`** ：

```bash
pip install cnocr
```



### 脚本使用

**cnstd** 包含了几个命令行命令，安装 **cnstd** 后即可使用。



#### 预测单个文件或文件夹中所有图片

使用命令 **`cnstd evaluate`** 预测单个文件或文件夹中所有图片，以下是使用说明：

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd evaluate -h
Usage: cnstd evaluate [OPTIONS]

Options:
  --backbone [mobilenetv3|resnet50_v1b]
                                  backbone model name
  --model_root_dir TEXT           模型所在的根目录
  --model_epoch INTEGER           model epoch
  -i, --img_dir TEXT              评估图片所在的目录或者单个图片文件路径
  --max_size INTEGER              图片预测时的最大尺寸（最好是32的倍数）。超过这个尺寸的图片会被等比例压缩到此尺寸
                                  [Default: 768]

  --pse_threshold FLOAT           threshold for pse [Default: 0.45]
  --pse_min_area INTEGER          min area for pse [Default: 100]
  --gpu INTEGER                   使用的GPU数量。默认值为-1，表示自动判断
  -o, --output_dir TEXT           输出结果存放的目录
  -h, --help                      Show this message and exit.
```



例如可以使用以下命令对图片 `examples/taobao.jpg`进行检测，并把检测结果存放在目录 `outputs`中：

```bash
cnstd evaluate -i examples/taobao.jpg -o outputs
```



具体使用也可参考文件 [Makefile](./Makefile) 。



#### 模型训练

使用命令 **`cnstd train`**  训练文本检测模型，以下是使用说明：

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd train -h
Usage: cnstd train [OPTIONS]

Options:
  --backbone [mobilenetv3|resnet50_v1b]
                                  backbone model name
  --pretrain_model_fp TEXT        初始化模型路径
  --gpu INTEGER                   使用的GPU数量。默认值为-1，表示自动判断
  --optimizer TEXT                optimizer for training [Default: Adam]
  --batch_size INTEGER            batch size for each device [Default: 4]
  --epoch INTEGER                 train epochs [Default: 50]
  --lr FLOAT                      learning rate [Default: 0.001]
  --momentum FLOAT                momentum [Default: 0.9]
  --wd FLOAT                      weight decay factor [Default: 0.0]
  --log_step INTEGER              隔多少步打印一次信息 [Default: 5]
  -r, --root_dir TEXT             数据所在的根目录，它与索引文件中指定的文件路径合并后获得最终的文件路径
  -i, --train_index_fp TEXT       存放训练数据的索引文件
  -o, --output_dir TEXT           模型输出的根目录 [Default: ~/.cnstd]
  -h, --help                      Show this message and exit.
```



具体使用可参考文件 [Makefile](./Makefile) 。



## 未来工作



* [ ] 进一步精简模型结构，降低模型大小。
* [ ] PSENet速度上还是比较慢，尝试更快的STD算法。
* [ ] 加入更多的训练数据。

