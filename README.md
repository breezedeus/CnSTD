# cnstd

**cnstd** 是 **Python 3** 下的**场景文字检测**（**Scene Text Detection**，简称**STD**）工具包，支持**中文**、**英文**等语言的文字检测，自带了多个训练好的检测模型，安装后即可直接使用。欢迎扫码加入QQ交流群：



![QQ群二维码](./docs/cnstd-qq.jpg)



自 **v1.0.0** 版本开始，**cnstd** 从之前基于 MXNet 实现转为基于 **PyTorch** 实现。新模型的训练合并了  **ICPR MTWI 2018**、**ICDAR RCTW-17** 和 **ICDAR2019-LSVT** 三个数据集，包括了 **`46447`** 个训练样本，和 **`1534`** 个测试样本。



相较于 V1.0.0， **V1.1.0** 的变化主要包括：

* bugfixes：修复了训练过程中发现的诸多问题；
* 检测主类 **`CnStd`** 初始化接口略有调整，去掉了参数 `model_epoch`；
* backbone 结构中加入了对 **ShuffleNet** 的支持；
* 优化了训练中的超参数取值，提升了模型检测精度；
* 提供了更多的预训练模型可供选择，最小模型降至 **7.5M** 文件大小。



如需要识别文本框中的文字，可以结合 **OCR** 工具包 **[cnocr](https://github.com/breezedeus/cnocr)** 一起使用。



## 示例

![文本检测示例](./docs/cases.png)





## 安装

嗯，安装真的很简单。

```bash
pip install cnstd
```



安装速度慢的话，可以指定国内的安装源，如使用豆瓣源：

```bash
pip install cnstd -i https://pypi.doubanio.com/simple
```



【注意】：

* 请使用 **Python3** (3.6以及之后版本应该都行)，没测过Python2下是否ok。
* 依赖 **opencv**，所以可能需要额外安装opencv。



## 已有模型

当前版本（**V1.1.0**）的文字检测模型使用的是 **[DBNet](https://github.com/MhLiao/DB)**，相较于 V0.1 使用的 [PSENet](https://github.com/whai362/PSENet) 模型， DBNet 的检测耗时几乎下降了一个量级，同时检测精度也得到了极大的提升。



目前包含以下已训练好的模型：

| 模型名称     | 参数规模 | 模型文件大小 | 测试集精度（IoU） | 平均推断耗时<br />（秒/张） | 下载方式 |
| ------------ |  -------- | -------- |------------ | -------- | -------- |
| db_resnet34 |  22.5 M | 86 M     | **0.7322**   | 3.11          | 自动 |
| db_resnet18 |  12.3 M | 47 M     | 0.7294      | 1.93          | 自动 |
| db_mobilenet_v3 |  4.2 M    | 16 M         | **0.7269**  | 1.76          | 自动 |
| db_mobilenet_v3_small |  2.0 M | 7.9 M | 0.7054 | 1.24 | 自动 |
| db_shufflenet_v2 |   4.7 M | 18 M | 0.7238 | 1.73 | 自动 |
| **db_shufflenet_v2_small** | 3.0 M | 12 M | 0.7190 | 1.29 | 自动 |
| db_shufflenet_v2_tiny | **1.9 M** | **7.5 M** | **0.7172** | **1.14** | [下载链接](https://mp.weixin.qq.com/s/fHPNoGyo72EFApVhEgR6Nw) |


> 上表耗时基于本地 Mac 获得，绝对值无太大参考价值，相对值可供参考。IoU的计算方式经过调整，仅相对值可供参考。



相对于两个基于 **ResNet** 的模型，基于 **MobileNet** 和 **ShuffleNet** 的模型体积更小，速度更快，建议在轻量级场景使用。



## 使用方法

首次使用 **cnstd** 时，系统会自动从 [贝叶智能](https://www.behye.com) 的CDN上下载zip格式的模型压缩文件，并存放于 `~/.cnstd`目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\cnstd`）。下载速度超快。下载后的zip文件代码会自动对其解压，然后把解压后的模型相关目录放于`~/.cnstd/1.1`目录中。



如果系统无法自动成功下载zip文件，则需要手动从 [百度云盘](https://pan.baidu.com/s/11_83ydAwJ1u8RnyyZtBKjw)（提取码为 `56ji`）下载对应的zip文件并把它存放于 `~/.cnstd/1.1`（Windows下为 `C:\Users\<username>\AppData\Roaming\cnstd\1.1`）目录中。模型也可从 **[cnstd-cnocr-models](https://github.com/breezedeus/cnstd-cnocr-models)** 中下载。放置好zip文件后，后面的事代码就会自动执行了。



### 图片预测

使用类 `CnStd` 进行场景文字的检测。类 `CnStd` 的初始化函数如下：

```python
class CnStd(object):
    """
    场景文字检测器（Scene Text Detection）。虽然名字中有个"Cn"（Chinese），但其实也可以轻松识别英文的。
    """

    def __init__(
        self,
        model_name: str = 'db_shufflenet_v2_small',
        *,
        auto_rotate_whole_image: bool = False,
        rotated_bbox: bool = True,
        context: str = 'cpu',
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        **kwargs,
    ):
```

其中的几个参数含义如下：

* `model_name`:  模型名称，即上面表格第一列中的值。默认为 **db_shufflenet_v2_small** 。
* `auto_rotate_whole_image`:  是否自动对整张图片进行旋转调整。默认为`False`。
* `rotated_bbox`:  是否支持检测带角度的文本框；默认为 `True`，表示支持；取值为 `False` 时，只检测水平或垂直的文本。
* `context`：预测使用的机器资源，可取值为字符串`cpu`、`gpu`、`cuda:0`。
* `model_fp`:  如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（`.ckpt`文件）。
* `root`: 模型文件所在的根目录。
  
  * Linux/Mac下默认值为 `~/.cnstd`，表示模型文件所处文件夹类似 `~/.cnstd/1.1/db_shufflenet_v2_small`。
  * Windows下默认值为 `C:\Users\<username>\AppData\Roaming\cnstd`。



每个参数都有默认取值，所以可以不传入任何参数值进行初始化：`std = CnStd()`。




文本检测使用类`CnOcr`的函数 **`detect()`**，以下是详细说明：



#### 类函数`CnStd.detect()`

```python
    def detect(
        self,
        img_list: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
            List[Union[str, Path, Image.Image, np.ndarray]],
        ],
        resized_shape: Tuple[int, int] = (768, 768),
        preserve_aspect_ratio: bool = True,
        min_box_size: int = 8,
        box_score_thresh: float = 0.3,
        batch_size: int = 20,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
```



**函数说明**：

函数输入参数包括：

-  `img_list`: 支持对单个图片或者多个图片（列表）的检测。每个值可以是图片路径，或者已经读取进来 `PIL.Image.Image` 或 `np.ndarray`,  格式应该是 `RGB` 3 通道，shape: `(height, width, 3)`, 取值范围：`[0, 255]`。
-  `resized_shape`:  `(height, width)`, 检测前，会先把原始图片 resize 到此大小。默认为 `(768, 768)`。
		注：其中取值必须都能整除 `32`。这个取值对检测结果的影响较大，可以针对自己的应用多尝试几组值，再选出最优值。例如 `(512, 768)`, `(768, 768)`, `(768, 1024)`等。
-  `preserve_aspect_ratio`: 对原始图片 resize 时是否保持高宽比不变。默认为 `True`。
-  `min_box_size`: 过滤掉高度或者宽度小于此值的文本框。默认为 `8`，也即高或者宽小于 `8` 的文本框会被过滤去掉。
-  `box_score_thresh`: 过滤掉得分低于此值的文本框。默认为 `0.3`。
-  `batch_size`: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `20`。
-  `kwargs`: 保留参数，目前未被使用。



函数输出类型为`list`，其中每个元素是一个字典，对应一张图片的检测结果。字典中包含以下 `keys`：

-  `rotated_angle`: `float`, 整张图片旋转的角度。只有 `auto_rotate_whole_image==True` 才可能非 `0`。

-  `detected_texts`: `list`, 每个元素存储了检测出的一个框的信息，使用词典记录，包括以下几个值：

	- `box`：检测出的文字对应的矩形框；4个 (`rotated_bbox==False`) 或者 5个 (`rotated_bbox==True`) 元素;
		
		- 4个元素时的含义：对应 `rotated_bbox==False`，取值为：`[xmin, ymin, xmax, ymax]` ;
		- 5个元素时的含义：对应 `rotated_bbox==True`，取值为：`[x, y, w, h, angle]`。
		
	- "score"：得分；`float` 类型；分数越高表示越可靠；
	
	- "croppped_img"：对应 "box" 中的图片patch（`RGB`格式），会把倾斜的图片旋转为水平。`np.ndarray`类型，`shape: (height, width, 3)`,  取值范围：`[0, 255]`；
		
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
from PIL import Image
from cnstd import CnStd

std = CnStd()
img_fp = 'examples/taobao.jpg'
img = Image.open(img_fp)
box_infos = std.detect(img)
```



### 识别检测框中的文字（OCR）

上面示例识别结果中"cropped_img"对应的值可以直接交由 **[cnocr](https://github.com/breezedeus/cnocr)** 中的 **`CnOcr`** 进行文字识别。如上例可以结合  **`CnOcr`** 进行文字识别：

```python
from cnstd import CnStd
from cnocr import CnOcr

std = CnStd()
cn_ocr = CnOcr()

box_infos = std.detect('examples/taobao.jpg')

for box_info in box_infos['detected_texts']:
    cropped_img = box_info['cropped_img']
    ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
    print('ocr result: %s' % str(ocr_out))
```



注：运行上面示例需要先安装  **[cnocr](https://github.com/breezedeus/cnocr)** ：

```bash
pip install cnocr
```



### 脚本使用

**cnstd** 包含了几个命令行工具，安装 **cnstd** 后即可使用。



#### 预测单个文件或文件夹中所有图片

使用命令 **`cnstd predict`** 预测单个文件或文件夹中所有图片，以下是使用说明：

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd predict -h
Usage: cnstd predict [OPTIONS]

  预测单个文件，或者指定目录下的所有图片

Options:
  -m, --model-name [db_resnet50|db_resnet34|db_resnet18|db_mobilenet_v3|db_mobilenet_v3_small|db_shufflenet_v2|db_shufflenet_v2_small|db_shufflenet_v2_tiny]
                                  模型名称。默认值为 `db_shufflenet_v2_small`
  --model-epoch INTEGER           model epoch。默认为 `None`，表示使用系统自带的预训练模型
  -p, --pretrained-model-fp TEXT  导入的训练好的模型，作为初始模型。默认为 `None`，表示使用系统自带的预训练模型
  -r, --rotated-bbox              是否检测带角度（非水平和垂直）的文本框。默认为 `True`
  --resized-shape TEXT            格式："height,width";
                                  预测时把图片resize到此大小再进行预测。两个值都需要是32的倍数。默认为
                                  `768,768`

  --box-score-thresh FLOAT        检测结果只保留分数大于此值的文本框。默认值为 `0.3`
  --preserve-aspect-ratio BOOLEAN
                                  resize时是否保留图片原始比例。默认值为 `True`
  --context TEXT                  使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`
  -i, --img-file-or-dir TEXT      输入图片的文件路径或者指定的文件夹
  -o, --output-dir TEXT           检测结果存放的文件夹。默认为 `./predictions`
  -h, --help                      Show this message and exit.
```



例如可以使用以下命令对图片 `examples/taobao.jpg`进行检测，并把检测结果存放在目录 `outputs`中：

```bash
cnstd predict -i examples/taobao.jpg -o outputs
```



具体使用也可参考文件 [Makefile](./Makefile) 。



#### 模型训练

使用命令 **`cnstd train`**  训练文本检测模型，以下是使用说明：

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd train -h
Usage: cnstd train [OPTIONS]

  训练文本检测模型

Options:
  -m, --model-name [db_resnet50|db_resnet34|db_resnet18|db_mobilenet_v3|db_mobilenet_v3_small|db_shufflenet_v2|db_shufflenet_v2_small|db_shufflenet_v2_tiny]
                                  模型名称。默认值为 `db_shufflenet_v2_small`
  -i, --index-dir TEXT            索引文件所在的文件夹，会读取文件夹中的 `train.tsv` 和 `dev.tsv` 文件
                                  [required]

  --train-config-fp TEXT          训练使用的json配置文件  [required]
  -r, --resume-from-checkpoint TEXT
                                  恢复此前中断的训练状态，继续训练
  -p, --pretrained-model-fp TEXT  导入的训练好的模型，作为初始模型。优先级低于 "--restore-training-
                                  fp"，当传入"--restore-training-fp"时，此传入失效

  -h, --help                      Show this message and exit.
```



具体使用可参考文件 [Makefile](./Makefile) 。



#### 模型转存

训练好的模型会存储训练状态，使用命令 **`cnstd resave`**  去掉与预测无关的数据，降低模型大小。

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd resave -h
Usage: cnstd resave [OPTIONS]

  训练好的模型会存储训练状态，使用此命令去掉预测时无关的数据，降低模型大小

Options:
  -i, --input-model-fp TEXT   输入的模型文件路径  [required]
  -o, --output-model-fp TEXT  输出的模型文件路径  [required]
  -h, --help                  Show this message and exit.
```





## 未来工作



* [x] 进一步精简模型结构，降低模型大小。
* [x] PSENet速度上还是比较慢，尝试更快的STD算法。
* [x] 加入更多的训练数据。
* [ ] 加入对文档结构与表格的检测