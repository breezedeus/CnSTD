# coding: utf-8

import os

from cnstd import CnStd
from cnstd.utils import imsave


def read_idx_file(idx_fp):
    img_label_pairs = []
    with open(idx_fp) as f:
        for line in f:
            img_fp, gt_fp = line.strip().split('\t')
            img_fp = img_fp.split('\\')[-1]
            img_label_pairs.append((img_fp, gt_fp))
    return img_label_pairs


def main():
    root_data_dir = '/Users/king/Documents/beiye-Ein/语料/ocr/From-CnOCR-Users/ocr'
    index_fp = os.path.join(root_data_dir, 'train.tsv')
    image_dir = os.path.join(root_data_dir, 'train_pic')
    out_index_fp = open(os.path.join(root_data_dir, 'train_cleaned.tsv'), 'w')
    out_image_dir = os.path.join(root_data_dir, 'train_pic_cleaned')
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    img_label_pairs = read_idx_file(index_fp)
    std_model_name = 'db_shufflenet_v2_small'
    std = CnStd(
        std_model_name,
        rotated_bbox=False,
        context='cpu',
    )
    resized_shape = (384, 384)

    num_success = 0
    num_total = len(img_label_pairs)
    for idx, (img_fp, label) in enumerate(img_label_pairs):
        if idx % 100 == 0:
            print(f'{idx=}, {num_success=}')
        std_out = std.detect(
            os.path.join(image_dir, img_fp),
            resized_shape=resized_shape,
            preserve_aspect_ratio=True,
            box_score_thresh=0.3,
        )
        # if img_fp == 'A_ISO_18.JPG':
        #     breakpoint()
        if len(std_out['detected_texts']) != 1:
            continue
        cropped_img = std_out['detected_texts'][0]['cropped_img']
        h, w = cropped_img.shape[:2]
        if w < 2.5 * h:
            continue

        imsave(cropped_img, os.path.join(out_image_dir, img_fp), normalized=False)
        out_index_fp.write(f'{img_fp}\t{label}\n')
        num_success += 1

    print(f'Totally, {num_total=}, {num_success=}.')
    out_index_fp.close()


if __name__ == '__main__':
    main()
