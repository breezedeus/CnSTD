# coding: utf-8
# 生成检测结果（json格式）文件，这个文件可以导入到label studio中，生成待标注的任务
from collections import OrderedDict
from glob import glob
import json
from argparse import ArgumentParser
import hashlib
import tqdm

from PIL import Image
import cv2

from cnstd import LayoutAnalyzer
from cnstd.utils import read_img


def to_json(total_width, total_height, box_type, x0, y0, w, h, _id):
    return {
        "original_width": total_width,
        "original_height": total_height,
        "image_rotation": 0,
        "value": {
            "x": x0,
            "y": y0,
            "width": w,
            "height": h,
            "rotation": 0,
            "rectanglelabels": [box_type],
        },
        "id": str(_id),
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "manual",
    }


def deduplicate_images(img_dir):
    def calculate_image_hash(image_path):
        # with open(img_fp, 'rb') as f:
        #     image_data = f.read()
        #     return hashlib.md5(image_data).hexdigest()
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        _, threshold = cv2.threshold(
            resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return sum([2 ** i for (i, v) in enumerate(threshold.flatten()) if v])

    img_fp_list = glob('{}/*g'.format(img_dir), recursive=True)
    print(f'{len(img_fp_list)} images found in {img_dir}')
    outs = OrderedDict()
    for img_fp in tqdm.tqdm(img_fp_list):
        img_hash = calculate_image_hash(img_fp)

        # 将特征值与文件名存储在字典中
        if img_hash not in outs:
            outs[img_hash] = img_fp
    print(f'{len(outs)} different images kept after deduplication')
    return list(outs.values())


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--img-dir', type=str, required=True, help='image directory'
    )

    parser.add_argument(
        '-t',
        '--model-type',
        type=str,
        default='yolov7',
        help='模型类型。当前支持 [`yolov7_tiny`, `yolov7`]',
    )
    parser.add_argument(
        '-p',
        '--model-fp',
        type=str,
        default='epoch_124-mfd.pt',
        help='使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型',
    )

    parser.add_argument(
        '-o',
        '--out-json-fp',
        type=str,
        default='prediction_results.json',
        help='output json file',
    )
    args = parser.parse_args()
    img_dir = args.img_dir

    analyzer = LayoutAnalyzer(
        model_name='mfd', model_type='yolov7', model_fp=args.model_fp
    )

    img_fp_list = deduplicate_images(img_dir)

    total_json = []
    num_boxes = 0
    for img_fp in tqdm.tqdm(img_fp_list):
        img0 = read_img(img_fp, return_type='Image')
        width, height = img0.size
        out = analyzer.analyze(img0, resized_shape=608)

        results = []
        for box_info in out:
            num_boxes += 1
            # box with 4 points to (x0, y0, w, h)
            box = box_info['box']
            w = box[2][0] - box[0][0]
            h = box[2][1] - box[0][1]
            info = to_json(
                width,
                height,
                box_info['type'],
                100 * box[0][0] / width,
                100 * box[0][1] / height,
                100 * w / width,
                100 * h / height,
                num_boxes,
            )
            results.append(info)

        predictions = [{"model_version": "one", "score": 0.5, "result": results}]
        data = {
            # "image": img_fp,
            "image": "/data/local-files/?d="
            + img_fp,
        }
        total_json.append({"data": data, "predictions": predictions})

    json.dump(total_json, open(args.out_json_fp, 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
