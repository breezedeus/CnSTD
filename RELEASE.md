# Release Notes

# Update 2020.07.01：发布 cnstd V0.1.1

`CnStd.detect()`加入输入参数 `kwargs`: 目前会使用到的keys有：
  * "height_border"，裁切图片时在高度上留出的边界比例，最终上下总共留出的边界大小为height * height_border; 默认为0.05；
  * "width_border"，裁切图片时在宽度上留出的边界比例，最终左右总共留出的边界大小为height * width_border; 默认为0.0；

bugfix:
  * 修复GPU下推断bug：https://github.com/breezedeus/cnstd/issues/3


# Update 2020.06.02：发布 cnstd V0.1.0

初次发布，主要功能：

* 利用PSENet进行场景文字检测（STD），支持两种backbone模型：`mobilenetv3` 和 `resnet50_v1b`。

