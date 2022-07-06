# Release Notes


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

