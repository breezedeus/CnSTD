# Release Notes

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

