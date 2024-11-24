<div align="center">
	<img src="./docs/logo.png" width="250px"/>
  <div>&nbsp;</div>

[![Downloads](https://static.pepy.tech/personalized-badge/cnstd?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/cnstd)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FCnSTD&label=Visitors&countColor=%23f5c791&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FCnSTD)
[![license](https://img.shields.io/github/license/breezedeus/cnstd)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/cnstd.svg)](https://badge.fury.io/py/cnstd)
[![forks](https://img.shields.io/github/forks/breezedeus/cnstd)](https://img.shields.io/github/forks/breezedeus/cnstd)
[![stars](https://img.shields.io/github/stars/breezedeus/cnstd)](https://github.com/breezedeus/cnocr)
![last-releast](https://img.shields.io/github/release-date/breezedeus/cnstd?style=plastic)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/cnstd)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

</div>

<div align="center">

[中文](./README.md) | English

</div>

# CnSTD

## Update 2024.11.24: Release V1.2.5

Major Changes:

* Integrated latest PPOCRv4 text detection functionality based on RapidOCR for faster inference
  * Added support for PP-OCRv4 detection models, including standard and server versions
  * Added support for PP-OCRv3 English detection model
* Optimized model download functionality with support for domestic mirrors

## Update 2024.06.16: Release V1.2.4

**Key Changes:**

* Support for YOLO Detector based on Ultralytics.

## Update 2023.06.30: Release V1.2.3

**Key Changes:**

* Retrained the **MFD YoloV7** model with new annotated data. The new model is now deployed on [P2T Web](https://p2t.behye.com). Details: [Pix2Text (P2T) New Formula Detection Model | Breezedeus.com](https://www.breezedeus.com/article/p2t-mfd-20230613).
* The previous MFD YoloV7 model is now available for Planet members. Details: [P2T YoloV7 Math Formula Detection Model Available for Planet Members | Breezedeus.com](https://www.breezedeus.com/article/p2t-yolov7-for-zsxq-20230619).
* Added some Label Studio related scripts in the [scripts](scripts) folder. These scripts use the MFD model to detect formulas in images and generate JSON files importable to Label Studio. Additionally, they convert JSON files exported from Label Studio to the format required for training the MFD model. Note that the MFD model training code is in the [yolov7](https://github.com/breezedeus/yolov7) (`dev` branch).

For more details: [RELEASE.md](./RELEASE.md).

---

**CnSTD** is a **Scene Text Detection** (STD) toolkit under **Python 3**, supporting text detection in languages such as **Chinese** and **English**. It comes with several pre-trained detection models ready for use upon installation. From **V1.2.1**, **CnSTD** includes a **Mathematical Formula Detection** (MFD) model, offering pre-trained models for detecting mathematical formulas (both inline `embedding` and standalone `isolated`).

Join our WeChat group by scanning the QR code:

<div align="center">
  <img src="https://huggingface.co/datasets/breezedeus/cnocr-wx-qr-code/resolve/main/wx-qr-code.JPG" alt="WeChat Group QR Code" width="300px"/>
</div>

The author also maintains a **Knowledge Planet** group [**CnOCR/CnSTD/P2T Private Group**](https://t.zsxq.com/FEYZRJQ). This group provides private materials related to CnOCR/CnSTD/P2T, including detailed training tutorials, unpublished models, solutions to challenges, and the latest research materials on OCR/STD.

Starting from **V1.0.0**, **CnSTD** transitioned from an MXNet-based implementation to a **PyTorch**-based implementation. The new model training incorporates the **ICPR MTWI 2018**, **ICDAR RCTW-17**, and **ICDAR2019-LSVT** datasets, comprising **46,447** training samples and **1,534** test samples.

Compared to previous versions, the new version features:

* Support for [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) detection models.
* Adjusted the expression of `box` in detection results to uniformly use `4` coordinate points.
* Bug fixes.

For text recognition within detected text boxes, use the **OCR** toolkit **[cnocr](https://github.com/breezedeus/cnocr)**.

## Examples

### Scene Text Detection (STD)

<div align="center">
  <img src="./docs/cases.png" alt="STD Example" width="700px"/>
</div>

### Mathematical Formula Detection (MFD)

The MFD model detects mathematical formulas in images, categorizing them as `embedding` for inline formulas and `isolated` for standalone formulas. The model is trained using the English [IBEM](https://zenodo.org/record/4757865) and Chinese [CnMFD_Dataset](https://github.com/breezedeus/CnMFD_Dataset) datasets.

<div align="center">
  <img src="./examples/mfd/out-zh4.jpg" alt="Chinese MFD Example" width="700px"/>
</div>  
<div align="center">
  <img src="./examples/mfd/out-zh5.jpg" alt="Chinese MFD Example" width="700px"/>
</div>
<div align="center">
  <img src="./examples/mfd/out-en2.jpg" alt="English MFD Example" width="700px"/>
</div>

### Layout Analysis

The layout analysis model identifies different layout elements in images. The model is trained using the [CDLA](https://github.com/buptlihang/CDLA) dataset and can recognize the following 10 layout elements:

| Text | Title | Figure | Figure Caption | Table | Table Caption | Header | Footer | Reference | Equation |
|------|-------|--------|----------------|-------|---------------|--------|--------|-----------|----------|

<div align="center">
  <img src="./examples/layout/out-zh.jpg" alt="Layout Analysis Example" width="700px"/>
</div>

## Installation

Installation is straightforward:

```bash
pip install cnstd
```

To use ONNX models (`model_backend=onnx`), use the following commands:

* For CPU:
  ```bash
  pip install cnstd[ort-cpu]
  ```
* For GPU:
  ```bash
  pip install cnstd[ort-gpu]
  ```
  * Note: If `onnxruntime` is already installed, uninstall it first (`pip uninstall onnxruntime`) before running the above commands.

For faster installation, specify a domestic source like Douban:

```bash
pip install cnstd -i https://mirrors.aliyun.com/pypi/simple
```

**Note:**

* Use **Python3** (version 3.6 or later).
* Dependency: **opencv**.

## Available STD Models

Starting from **V1.2**, CnSTD includes two types of models: 1) models trained by CnSTD, usually available in PyTorch and ONNX versions; 2) pre-trained models from other OCR engines, converted to ONNX for use in CnSTD.

Downloadable models are available in the [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) project.

### 1. CnSTD Trained Models

The current version (Since **V1.1.0**) uses [**DBNet**](https://github.com/MhLiao/DB), which significantly reduces detection time and improves accuracy compared to the previous [PSENet](https://github.com/whai362/PSENet) model.

Available models:

| Model Name                | Parameters | File Size | Test Accuracy (IoU) | Average Inference Time (s/img) | Download Link |
|---------------------------|------------|-----------|---------------------|-------------------------------|---------------|
| db_resnet34               | 22.5 M     | 86 M      | **0.7322**          | 3.11                          | Automatic     |
| db_resnet18               | 12.3 M     | 47 M      | 0.7294              | 1.93                          | Automatic     |
| db_mobilenet_v3           | 4.2 M      | 16 M      | **0.7269**          | 1.76                          | Automatic     |
| db_mobilenet_v3_small     | 2.0 M      | 7.9 M     | 0.7054              | 1.24                          | Automatic     |
| db_shufflenet_v2          | 4.7 M      | 18 M      | 0.7238              | 1.73                          | Automatic     |
| db_shufflenet_v2_small    | 3.0 M      | 12 M      | 0.7190              | 1.29                          | Automatic     |
| db_shufflenet_v2_tiny     | **1.9 M**  | **7.5 M** | **0.7172**          | **1.14**                      | [Download Link](https://mp.weixin.qq.com/s/fHPNoGyo72EFApVhEgR6Nw) |

> The above times are based on a local Mac. Absolute values may not be very useful, but relative values are for reference. IoU calculation has been adjusted, so only relative values are for reference.

Models based on **MobileNet** and **ShuffleNet** are smaller and faster than those based on **ResNet**, making them suitable for lightweight scenarios.

### 2. External Models

The following models are ONNX versions from [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR), so they do not depend on **PaddlePaddle** and do not support further fine-tuning on specific domain data. These models support vertical text detection.

| `model_name`    | PyTorch Version | ONNX Version | Supported Languages | Model File Size |
|-----------------|-----------------|--------------|---------------------|-----------------|
| ch_PP-OCRv4_det | X               | √            | Chinese, English, Numbers | 4.5 M        |
| ch_PP-OCRv4_det_server | X               | √            | Chinese, English, Numbers | 108 M        |
| ch_PP-OCRv3_det | X               | √            | Chinese, English, Numbers | 2.2 M        |
| en_PP-OCRv3_det | X               | √            | **English**, Numbers | 2.3 M          |

For more models, refer to [PaddleOCR/models_list

.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/models_list.md). If you need models for other languages (e.g., Japanese, Korean), you can suggest them in the **Knowledge Planet** [**CnOCR/CnSTD Private Group**](https://t.zsxq.com/FEYZRJQ).

## Usage

When using **CnSTD** for the first time, the system will automatically download the model zip file and store it in the `~/.cnstd` directory (default path for Windows is `C:\Users\<username>\AppData\Roaming\cnstd`). The download is fast, and the code will automatically unzip and place the extracted model in the `~/.cnstd/1.2` directory.

If the system cannot automatically download the zip file, manually download it from [Baidu Cloud](https://pan.baidu.com/s/1zDMzArCDrrXHWL0AWxwYQQ?pwd=nstd) (password: `nstd`) and place it in the `~/.cnstd/1.2` (Windows: `C:\Users\<username>\AppData\Roaming\cnstd\1.2`) directory. You can also download the model from [cnstd-cnocr-models](https://huggingface.co/breezedeus/cnstd-cnocr-models). Once the zip file is in place, the code will automatically handle the rest.

### Scene Text Detection (STD)

Use the `CnStd` class for scene text detection. The `CnStd` class initializer is as follows:

```python
class CnStd(object):
    """
    Scene Text Detection (STD) class. Despite the "Cn" (Chinese) in the name, it can also detect English text.
    """

    def __init__(
        self,
        model_name: str = 'ch_PP-OCRv4_det',
        *,
        auto_rotate_whole_image: bool = False,
        rotated_bbox: bool = True,
        context: str = 'cpu',
        model_fp: Optional[str] = None,
        model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        root: Union[str, Path] = data_dir(),
        use_angle_clf: bool = False,
        angle_clf_configs: Optional[dict] = None,
        **kwargs,
    ):
```

Key parameters:

* `model_name`: Model name, corresponding to the first column in the model table. Default is **ch_PP-OCRv4_det**.
* `auto_rotate_whole_image`: Automatically adjust the rotation of the entire image. Default is `False`.
* `rotated_bbox`: Support detection of angled text boxes; Default is `True`. If `False`, only horizontal or vertical text is detected.
* `context`: Resource for prediction, can be `cpu`, `gpu`, or `cuda:0`.
* `model_fp`: Custom model file (`.ckpt`). Default is `None`, using the system's pre-trained model.
* `model_backend`: `str`: 'pytorch' or 'onnx'. Indicates the backend for prediction. Default is `onnx`.
* `root`: Root directory for model files. 
  * Default for Linux/Mac is `~/.cnstd`.
  * Default for Windows is `C:\Users\<username>\AppData\Roaming\cnstd`.
* `use_angle_clf`: Use angle classification model to adjust detected text boxes (useful for boxes that may be rotated 180 degrees). Default is `False`.
* `angle_clf_configs`: Parameters for the angle classification model, mainly:
  - `model_name`: Default is 'ch_ppocr_mobile_v2.0_cls'.
  - `model_fp`: Custom model file (`.onnx`). Default is `None`.

All parameters have default values, so you can initialize without any parameters: `std = CnStd()`.

To detect text, use the **`detect()`** method of the `CnStd` class. Detailed explanation:

#### `CnStd.detect()`

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
        resized_shape: Union[int, Tuple[int, int]] = (768, 768),
        preserve_aspect_ratio: bool = True,
        min_box_size: int = 8,
        box_score_thresh: float = 0.3,
        batch_size: int = 20,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
```

**Function Explanation**:

Input parameters:

- `img_list`: Supports single or multiple images (list). Each value can be an image path, or already read `PIL.Image.Image` or `np.ndarray`, in `RGB` format with shape `(height, width, 3)`, value range: `[0, 255]`.
- `resized_shape`: `int` or `tuple`, `tuple` means `(height, width)`, `int` means both height and width are this value. Default is `(768, 768)`.
- `preserve_aspect_ratio`: Keep original aspect ratio when resizing. Default is `True`.
- `min_box_size`: Filter out text boxes with height or width smaller than this value. Default is `8`.
- `box_score_thresh`: Filter out text boxes with a score lower than this value. Default is `0.3`.
- `batch_size`: Number of images per batch when processing many images. Default is `20`.
- `kwargs`: Reserved parameters, currently unused.

Output type is `list`, where each element is a dictionary representing the detection result for an image. The dictionary includes the following keys:

- `rotated_angle`: `float`, rotation angle of the entire image. Only non-zero if `auto_rotate_whole_image==True`.
- `detected_texts`: `list`, each element is a dictionary with information about a detected text box:
  - `box`: Detected text box; `np.ndarray`, shape: `(4, 2)`, coordinates of the box's 4 points `(x, y)`.
  - `score`: Score; `float` type; higher score means more reliable.
  - `cropped_img`: Image patch corresponding to "box" (`RGB` format), with tilted images rotated to horizontal. `np.ndarray`, shape: `(height, width, 3)`, value range: `[0, 255]`.
  - Example:
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

#### Example

```python
from cnstd import CnStd
std = CnStd()
box_info_list = std.detect('examples/taobao.jpg')
```

Or:

```python
from PIL import Image
from cnstd import CnStd

std = CnStd()
img_fp = 'examples/taobao.jpg'
img = Image.open(img_fp)
box_infos = std.detect(img)
```

### Text Recognition within Detected Text Boxes (OCR)

The `cropped_img` values in the detection result can be recognized using the **[cnocr](https://github.com/breezedeus/cnocr)** `CnOcr` class. For example:

```python
from cnstd import CnStd
from cnocr import CnOcr

std = CnStd()
cn_ocr = CnOcr()

box_infos = std.detect('examples/taobao.jpg')

for box_info in box_infos['detected_texts']:
    cropped_img = box_info['cropped_img']
    ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
    print('ocr result: %s' % str(ocr_res))
```

To run the example, first install **[cnocr](https://github.com/breezedeus/cnocr)**:

```bash
pip install cnocr
```

### Mathematical Formula Detection (MFD) and Layout Analysis

Both MFD and Layout Analysis detect elements of interest in images using YOLOv7-based models. In CnSTD, they are implemented in the same class `LayoutAnalyzer`, with the difference being the data used for training.

> The training code for these models is in [yolov7](https://github.com/breezedeus/yolov7) (forked from [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)). Thanks to the original author.

The `LayoutAnalyzer` class initializer:

```python
class LayoutAnalyzer(object):
    def __init__(
        self,
        model_name: str = 'mfd',  # 'layout' or 'mfd'
        *,
        model_type: str = 'yolov7_tiny',  # current support ['yolov7_tiny', 'yolov7']
        model_backend: str = 'pytorch',
        model_categories: Optional[List[str]] = None,
        model_fp: Optional[str] = None,
        model_arch_yaml: Optional[str] = None,
        root: Union[str,

 Path] = data_dir(),
        device: str = 'cpu',
        **kwargs,
    )
```

Key parameters:

* `model_name`: String, model type. Options: 'mfd' for mathematical formula detection; 'layout' for layout analysis. Default: 'mfd'.
* `model_type`: String, model type. Current options: 'yolov7_tiny' and 'yolov7'. Default: 'yolov7_tiny'. 'yolov7' models are currently only available to Planet members. Details: [P2T YoloV7 Mathematical Formula Detection Model Available for Planet Members | Breezedeus.com](https://www.breezedeus.com/article/p2t-yolov7-for-zsxq-20230619).
* `model_backend`: String, backend. Current support: 'pytorch'. Default: 'pytorch'.
* `model_categories`: Detection category names. Default: None, determined based on `model_name`.
* `model_fp`: Path to model file. Default: `None`, using default path.
* `model_arch_yaml`: Path to architecture file, e.g., 'yolov7-mfd.yaml'. Default: None, auto-selected.
* `root`: String or `Path`, root directory for model files. 
  * Default for Linux/Mac is `~/.cnstd`.
  * Default for Windows is `C:/Users/<username>/AppData/Roaming/cnstd`.
* `device`: String, device for running the model, options: 'cpu' or 'gpu'. Default: 'cpu'.
* `kwargs`: Additional parameters.

#### `LayoutAnalyzer.analyze()`

Analyze specified images (or list of images) for layout analysis.

```python
def analyze(
    self,
    img_list: Union[
        str,
        Path,
        Image.Image,
        np.ndarray,
        List[Union[str, Path, Image.Image, np.ndarray]],
    ],
    resized_shape: Union[int, Tuple[int, int]] = 700,
    box_margin: int = 2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
```

**Function Explanation**:

Input parameters:

* `img_list`: `str` or `list`, path(s) to image(s) to be analyzed.
* `resized_shape`: `int` or `tuple` (H, W), resize image to this size for analysis. Default: `700`.
* `box_margin`: `int`, pixel margin to expand detected content boxes. Default: `2`.
* `conf_threshold`: `float`, confidence threshold. Default: `0.25`.
* `iou_threshold`: `float`, IOU threshold for NMS. Default: `0.45`.
* `kwargs`: Additional parameters.

Output is a `list`, where each element represents an identified layout element, including:

* `type`: Type of layout element, options from `self.categories`.
* `box`: Bounding box coordinates; `np.ndarray`, shape: (4, 2).
* `score`: Confidence score.

#### Example

```python
from cnstd import LayoutAnalyzer
img_fp = 'examples/mfd/zh5.jpg'
analyzer = LayoutAnalyzer('mfd')
out = analyzer.analyze(img_fp, resized_shape=700)
print(out)
```

### Using Scripts

**cnstd** includes several command-line tools, available after installation.

#### STD Prediction for a Single File or Directory

Use the `cnstd predict` command to predict text in a single file or all images in a directory. Usage:

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd predict -h
Usage: cnstd predict [OPTIONS]

  Predict text in a single file or all images in a directory.

Options:
  -m, --model-name [ch_PP-OCRv2_det|ch_PP-OCRv3_det|ch_PP-OCRv4_det|ch_PP-OCRv4_det_server|db_mobilenet_v3|db_mobilenet_v3_small|db_resnet18|db_resnet34|db_shufflenet_v2|db_shufflenet_v2_small|db_shufflenet_v2_tiny|en_PP-OCRv3_det]
                                  Model name. Default: db_shufflenet_v2_small.
  -b, --model-backend [pytorch|onnx]
                                  Model type. Default: `onnx`.
  -p, --pretrained-model-fp TEXT  Pre-trained model. Default: `None`.
  -r, --rotated-bbox              Detect angled text boxes. Default: `True`.
  --resized-shape TEXT            Format: "height,width". Resize image to this size for prediction. Values should be multiples of 32. Default: `768,768`.
  --box-score-thresh FLOAT        Filter out text boxes with a score lower than this value. Default: `0.3`.
  --preserve-aspect-ratio BOOLEAN Preserve original aspect ratio when resizing. Default: `True`.
  --context TEXT                  Use `cpu`, `gpu`, or specific gpu (e.g., `cuda:0`). Default: `cpu`.
  -i, --img-file-or-dir TEXT      Path to image file or directory.
  -o, --output-dir TEXT           Directory for prediction results. Default: `./predictions`.
  -h, --help                      Show this message and exit.
```

Example to detect text in `examples/taobao.jpg` and save results to `outputs`:

```bash
cnstd predict -i examples/taobao.jpg -o outputs
```

See the [Makefile](./Makefile) for more usage.

#### MFD or Layout Analysis for a Single File

Use the `cnstd analyze` command for MFD or Layout Analysis results for a single file. Usage:

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd analyze -h
Usage: cnstd analyze [OPTIONS]

  Perform MFD or Layout Analysis on a given image.

Options:
  -m, --model-name TEXT           Model type. `mfd` for mathematical formula detection, `layout` for layout analysis. Default: `mfd`.
  -t, --model-type TEXT           Model type. Current options: [`yolov7_tiny`, `yolov7`].
  -b, --model-backend [pytorch|onnx]
                                  Model backend. Current support: `pytorch`.
  -c, --model-categories TEXT     Model detection categories (comma-separated). Default: None.
  -p, --model-fp TEXT             Pre-trained model. Default: `None`.
  -y, --model-arch-yaml TEXT      Path to model configuration file.
  --device TEXT                   `cuda` device, e.g., `0` or `0,1,2,3` or `cpu`.
  -i, --img-fp TEXT               Path to image or directory.
  -o, --output-fp TEXT            Output path for analysis results. Default: `None`, saved in current folder with `out-` prefix.
  --resized-shape INTEGER         Resize image to this size for analysis. Default: `608`.
  --conf-thresh FLOAT             Confidence threshold. Default: `0.25`.
  --iou-thresh FLOAT              IOU threshold for NMS. Default: `0.45`.
  -h, --help                      Show this message and exit.
```

Example for MFD on `examples/mfd/zh.jpg` and save results to `out-zh.jpg`:

```bash
(venv) ➜  cnstd analyze -m mfd --conf-thresh 0.25 --resized-shape 800 -i examples/mfd/zh.jpg -o out-zh.jpg
```

See the [Makefile](./Makefile) for more usage.

#### Model Training

Use the `cnstd train` command to train text detection models. Usage:

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd train -h
Usage: cnstd train [OPTIONS]

  Train text detection models.

Options:
  -m, --model-name [db_resnet50|db_resnet34|db_resnet18|db_mobilenet_v3|db_mobilenet_v3_small|db_shufflenet_v2|db_shufflenet_v2_small|db_shufflenet_v2_tiny]
                                  Model name. Default: `db_shufflenet_v2_small`.
  -i, --index-dir TEXT            Directory for index files containing `train.tsv` and `dev.tsv`. [required]
  --train-config-fp TEXT          JSON configuration file for training. [required]
  -r, --resume-from-checkpoint TEXT
                                  Resume training from a checkpoint.
  -p, --pretrained-model-fp TEXT  Pre-trained model to use as initial model. Lower priority than `--restore-training-fp`.
  -h, --help                      Show this message and exit.
```

See the [Makefile](./Makefile) for more usage.

#### Model Resaving

Use the `cnstd resave` command to resave trained models, removing unnecessary data and reducing size.

```bash
(venv) ➜  cnstd git:(master) ✗ cnstd resave -h
Usage: cnstd resave [OPTIONS]

  Resave trained models, removing unnecessary data and reducing size.

Options:
  -i, --input-model-fp TEXT   Path to input model file. [required]
  -o, --output-model-fp TEXT  Path to output model file. [required]
  -h

, --help                  Show this message and exit.
```

## Future Work

* [x] Further simplify the model structure, reducing model size.
* [x] Explore faster STD algorithms.
* [x] Add more training data.
* [x] Support external models.
* [x] Add MFD and Layout Analysis models.
* [ ] Add document structure and table detection.

## Buy the Author a Coffee

Open source is challenging. If this project helps you, consider [buying the author a coffee ☕️](https://cnocr.readthedocs.io/zh/latest/buymeacoffee/).

---

Official repository: [https://github.com/breezedeus/cnstd](https://github.com/breezedeus/cnstd).
