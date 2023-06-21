# coding: utf-8
# 把 Label Studio 导出的JSON文件，转成训练 YoloV7 MFD模型所需的格式
import os
import json
from argparse import ArgumentParser


LABEL_MAPPINGS = {'embedding': 0, 'isolated': 1}


def bboxs_to_file(bboxs, out_fp):
    with open(out_fp, 'w') as f:
        for bbox in bboxs:
            bbox_str = ' '.join(map(str, bbox))
            f.write(f'{bbox_str}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--anno-json-fp-list',
        nargs='+',
        type=str,
        help='从Label Studio导出的标注结果JSON文件，可以有多个文件',
    )
    parser.add_argument(
        '--index-prefix', type=str, default='images', help='输出的索引文件中，每行图片的路径会加上这个前缀',
    )
    parser.add_argument(
        '--out-labels-dir',
        type=str,
        default='labels',
        help='输出的labels所在的文件夹，应该是 "**/**/labels/**" 这种格式',
    )
    parser.add_argument(
        '--out-index-fp', type=str, default='train.txt', help='输出的YoloV7 MFD格式的标注文件'
    )
    args = parser.parse_args()
    # img_root_dir = os.path.join(args.out_root_dir, 'images')
    # label_root_dir = os.path.join(args.out_root_dir, 'labels')
    label_root_dir = args.out_labels_dir

    fp_list = []
    for json_fp in args.anno_json_fp_list:
        ori_contents = json.load(open(json_fp))
        for info in ori_contents:
            fp_url = info['data']['image']
            remove_len = len(r'/data/local-files/?d=')
            fp = fp_url[remove_len:]
            annotations = info['annotations'][0].get('result', [])
            if not annotations:
                continue

            fn = os.path.basename(fp)
            # fp_list.append(os.path.join(img_root_dir, fp))
            fp_list.append(os.path.join(args.index_prefix, fn))

            # label_dir = os.path.join(label_root_dir, os.path.dirname(fp))
            label_dir = label_root_dir
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            label_fp = os.path.join(label_dir, fn.rsplit('.', maxsplit=1)[0] + '.txt')

            label_infos = []
            for annotation in annotations:
                value = annotation['value']
                x, y = value['x'], value['y']
                width, height = value['width'], value['height']
                # to [x0, y0, x1, y1, x2, y2, x3, y3]
                bbox = [x, y, x + width, y, x + width, y + height, x, y + height]
                bbox = [v * 0.01 for v in bbox]

                label = value['rectanglelabels'][0]
                label_id = LABEL_MAPPINGS[label]

                label_infos.append([label_id] + bbox)
            bboxs_to_file(label_infos, label_fp)

    with open(args.out_index_fp, 'w') as f:
        for fp in fp_list:
            f.write(fp + '\n')


if __name__ == '__main__':
    main()
