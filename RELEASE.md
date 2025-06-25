# Release Notes

## Update 2025.06.25：发布 V1.2.6

Major Changes:

* Integrated the latest PPOCRv5 text detection functionality based on RapidOCR for even faster inference speed
  * Added support for PP-OCRv5 detection models: `ch_PP-OCRv5_det` and `ch_PP-OCRv5_det_server`
* Fixed some known bugs

主要变更：

* 基于 RapidOCR 集成 PPOCRv5 最新版文本检测功能，提供更快的推理速度
  * 新增支持 PP-OCRv5 检测模型：`ch_PP-OCRv5_det` 和 `ch_PP-OCRv5_det_server`
* 修复部分已知 bug

## Update 2024.12.08：发布 V1.2.5.2

Bug Fixes:

* Fix compatibility issue of setting environment variables on Windows systems
* Use subprocess.run instead of os.system for better cross-platform support

Bug Fixes:

* 修复在 Windows 系统下设置环境变量的兼容性问题
* 使用 subprocess.run 替代 os.system 以提供更好的跨平台支持

## Update 2024.11.30：发布 V1.2.5.1 

Major Changes:

* en_PP-OCRv3_det still uses the previous version and does not use RapidDetector

Bug Fixes:

* en_PP-OCRv3_det 依旧使用之前的版本，不使用 RapidDetector

## Update 2024.11.24：发布 V1.2.5

Major Changes:

* Integrated latest PPOCRv4 text detection functionality based on RapidOCR for faster inference
  * Added support for PP-OCRv4 detection models, including standard and server versions
  * Added support for PP-OCRv3 English detection model
* Optimized model download functionality with support for domestic mirrors

主要变更：

* 基于 RapidOCR 集成 PPOCRv4 最新版文本检测功能，提供更快的推理速度
  * 新增支持 PP-OCRv4 检测模型，包括标准版和服务器版
  * 新增支持 PP-OCRv3 英文检测模型
* 优化模型下载功能，支持从国内镜像下载模型文件

# Update 2024.06.22：发布 V1.2.4.2

Major Changes:

* Added a new parameter `static_resized_shape` when initializing `YoloDetector`, which is used to resize the input image to a fixed size. Some formats of models require fixed-size input images during inference, such as `CoreML`.

主要变更：

* `YoloDetector` 初始化时加入了参数 `static_resized_shape`, 用于把输入图片 resize 为固定大小。某些格式的模型在推理时需要固定大小的输入图片，如 `CoreML`。

# Update 2024.06.17：发布 V1.2.4.1

Major Changes:

* Fixed a bug in the `detect` method of `YoloDetector`: when the input is a single file, the output is not a double-layer nested list.

主要变更：

* 修复了 `YoloDetector` 中 `detect` 方法的一个bug：输入为单个文件时，输出不是双层嵌套的 list。

# Update 2024.06.16：发布 V1.2.4

Major Changes:

* Support for YOLO Detector based on Ultralytics.


主要变更：

* 支持基于 Ultralytics 的 YOLO Detector。

# Update 2024.04.10：发布 V1.2.3.6

主要变更：

* CN OSS 不可用了，默认下载模型地址由 `CN` 改为 `HF`。

# Update 2023.10.09：发布 V1.2.3.5

主要变更：

* 支持基于环境变量 `CNSTD_DOWNLOAD_SOURCE` 的取值，来决定不同的模型下载路径。
* `LayoutAnalyzer` 中增加了参数 `model_categories` 和 `model_arch_yaml`，用于指定模型的类别名称列表和模型架构。

# Update 2023.09.23：发布 V1.2.3.4

主要变更：
* 增加了对 `onnxruntine` (ORT) 新版的兼容：`InferenceSession` 中显式提供了 `providers` 参数。
* `setup.py` 中去除对 `onnxruntime` 的依赖，改为在 `extras_require` 中按需指定：
  * `cnstd[ort-cpu]`：`onnxruntime`；
  * `cnstd[ort-gpu]`: `onnxruntime-gpu`。

# Update 2023.09.21：发布 V1.2.3.3

主要变更：
* 画图颜色优先使用固定的颜色组。
* 下载模型时支持设定环境变量 `HF_TOKEN`，以便从private repos中下载模型。

# Update 2023.07.02：发布 V1.2.3.2

主要变更：
* 修复参数 `device` 的取值bug，感谢 @Shadow-Alex 。

# Update 2023.06.30：发布 V1.2.3.1

主要变更：
* 修复比例转换后检测框可能出界的问题。

# Update 2023.06.30：发布 V1.2.3

主要变更：
* 修复了模型文件自动下载的功能。HuggingFace似乎对下载文件的逻辑做了调整，导致之前版本的自动下载失败，当前版本已修复。但由于HuggingFace国内被墙，国内下载仍需 **梯子（VPN）**。
* 更新了各个依赖包的版本号。

# Update 2023.06.20：

主要变更：
* 基于新标注的数据，重新训练了 **MFD YoloV7** 模型，目前新模型已部署到 [P2T网页版](https://p2t.behye.com) 。具体说明见：[Pix2Text (P2T) 新版公式检测模型 | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230613) 。
* 之前的 MFD YoloV7 模型已开放给星球会员下载，具体说明见：[P2T YoloV7 数学公式检测模型开放给星球会员下载 | Breezedeus.com](https://www.breezedeus.com/article/p2t-yolov7-for-zsxq-20230619) 。
* 增加了一些Label Studio相关的脚本，见 [scripts](scripts) 。如：利用 CnSTD 自带的 MFD 模型对目录中的图片进行公式检测后生成可导入到Label Studio中的JSON文件；以及，Label Studio标注后把导出的JSON文件转换成训练 MFD 模型所需的数据格式。注意，MFD 模型的训练代码在 [yolov7](https://github.com/breezedeus/yolov7) （`dev` branch）中。

# Update 2023.02.19：发布 V1.2.2

主要变更：
* MFD训练了参数更多精度更高的模型，供 [P2T网页版](https://p2t.behye.com) 使用。
* 优化了检测出的boxes的排序算法，使得boxes的顺序更加符合人类的阅读习惯。

# Update 2023.02.01：发布 V1.2.1

主要变更：
* 支持基于 **YOLOv7** 的 **数学公式检测**（**Mathematical Formula Detection**，简称**MFD**）和 **版面分析**（**Layout Analysis**）模型，并提供预训练好的模型可直接使用。
* 修复了不兼容 Numpy>=1.24 的bug。

# Update 2022.07.07：发布 cnstd V1.2

主要变更：
* 加入了对 [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) 检测模型的支持；
* 部分调整了检测结果中 `box` 的表达方式，统一为 `4` 个点的坐标值；
* 修复了已知bugs。
 

# Update 2022.05.27：发布 cnstd V1.1.2

主要变更：
* 兼容 `opencv-python >=4.5.2`，修复图片反转问题和画图报错问题。



# Update 2021.09.20：发布 cnstd V1.1.0

相较于 V1.0.0， **V1.1.0** 的变化主要包括：

* bugfixes：修复了训练过程中发现的诸多问题；
* 检测主类 **`CnStd`** 初始化接口略有调整，去掉了参数 `model_epoch`；
* backbone 结构中加入了对 **ShuffleNet** 的支持；
* 优化了训练中的超参数取值，提升了模型检测精度；
* 提供了更多的预训练模型可供选择，最小模型降至 **7.5M** 文件大小。



# Update 2021.08.26：发布 cnstd V1.0.0

* MXNet 越来越小众化，故从基于 MXNet 的实现转为基于 **PyTorch** 的实现；
* 检测速度得到极大提升，耗时几乎下降了一个量级；
* 检测精度也得到较大的提升；
* 实用性增强；检测接口中提供了更灵活的参数，不同应用场景可以尝试使用不同的参数以获得更好的检测效果；
* 提供了更丰富的预训练模型，开箱即用。




# Update 2020.07.01：发布 cnstd V0.1.1

`CnStd.detect()`加入输入参数 `kwargs`: 目前会使用到的keys有：
  * "height_border"，裁切图片时在高度上留出的边界比例，最终上下总共留出的边界大小为height * height_border; 默认为0.05；
  * "width_border"，裁切图片时在宽度上留出的边界比例，最终左右总共留出的边界大小为height * width_border; 默认为0.0；

bugfix:
  * 修复GPU下推断bug：https://github.com/breezedeus/cnstd/issues/3




# Update 2020.06.02：发布 cnstd V0.1.0

初次发布，主要功能：

* 利用PSENet进行场景文字检测（STD），支持两种backbone模型：`mobilenetv3` 和 `resnet50_v1b`。
