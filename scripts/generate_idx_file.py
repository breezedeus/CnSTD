import os
import glob
import json
from itertools import chain


def generate_icdar2015_idx_pairs(
    img_dir, img_prefix_dir, label_prefix_dir, label_prefix_fn=''
):
    imglst = glob.glob1(img_dir, '*g')
    imglst.sort()
    idx_pairs = []
    for img in imglst:
        img_fp = os.path.join(img_prefix_dir, img)
        name = img.rsplit('.', maxsplit=1)[0]
        label_fp = os.path.join(label_prefix_dir, label_prefix_fn + name + '.txt')
        idx_pairs.append((img_fp, label_fp))

    print(f'{len(idx_pairs)} pairs are generated')
    return idx_pairs


def generate_icpr_mtwi_2018_idx_pairs(img_dir, img_prefix_dir, label_prefix_dir):
    imglst = glob.glob1(img_dir, '*g')
    imglst.sort()
    idx_pairs = []
    for img in imglst:
        img_fp = os.path.join(img_prefix_dir, img)
        name = img.rsplit('.', maxsplit=1)[0]
        label_fp = os.path.join(label_prefix_dir, name + '.txt')
        idx_pairs.append((img_fp, label_fp))

    print(f'{len(idx_pairs)} pairs are generated')
    return idx_pairs


def save_idx_file(idx_pairs, output_fp):
    with open(output_fp, 'w') as f:
        for one_pair in idx_pairs:
            f.write('\t'.join(one_pair) + '\n')


def icdar2015():
    img_dir = 'data/icdar2015/train/images'
    img_prefix_dir = 'icdar2015/train/images'
    label_prefix_dir = 'icdar2015/train/gts'
    label_prefix_fn = 'gt_'
    idx_pairs = generate_icdar2015_idx_pairs(
        img_dir, img_prefix_dir, label_prefix_dir, label_prefix_fn
    )
    save_idx_file(idx_pairs, 'data/icdar2015/train.tsv')


def icpr_mtwi_2018():
    for i in ('1000', '9000'):
        img_dir = '/home/ein/jinlong/std_data/ICPR-MTWI-2018/train/image_%s' % i
        img_prefix_dir = 'ICPR-MTWI-2018/train/image_%s' % i
        label_prefix_dir = 'ICPR-MTWI-2018/train/txt_%s' % i
        idx_pairs = generate_icpr_mtwi_2018_idx_pairs(
            img_dir, img_prefix_dir, label_prefix_dir
        )
        save_idx_file(
            idx_pairs, '/home/ein/jinlong/std_data/ICPR-MTWI-2018/train_%s.tsv' % i
        )


def icdar_rctw_2017():
    img_dir = '/home/ein/jinlong/std_data/ICDAR-RCTW-2017/train_images'
    img_prefix_dir = 'ICDAR-RCTW-2017/train_images'
    label_prefix_dir = 'ICDAR-RCTW-2017/train_gts'
    idx_pairs = generate_icpr_mtwi_2018_idx_pairs(
        img_dir, img_prefix_dir, label_prefix_dir
    )
    save_idx_file(idx_pairs, '/home/ein/jinlong/std_data/ICDAR-RCTW-2017/train.tsv')


def generate_icpr_2019_lstv_idx_pairs(
    label_json_fp, out_label_dir, img_prefix_dir, label_prefix_dir
):
    idx_pairs = []
    labels = json.load(open(label_json_fp))
    for fname, info_list in labels.items():
        new_info = []
        for box_info in info_list:
            if len(box_info['points']) != 4:
                continue
            box = list(chain(*box_info['points']))

            text = box_info['transcription']
            if box_info['illegibility']:
                text = '###'

            box.append(text)
            new_info.append(list(map(str, box)))

        if new_info:
            label_fn = f'{fname}.txt'
            label_fp = os.path.join(out_label_dir, fname)
            with open(label_fp, 'w') as f:
                for line in new_info:
                    f.write(','.join(line) + '\n')

            img_fn = f'{fname}.jpg'
            img_fp = os.path.join(img_prefix_dir, img_fn)
            label_fp = os.path.join(label_prefix_dir, label_fn)
            idx_pairs.append((img_fp, label_fp))
    return idx_pairs


def icdar_2019_lstv():
    label_json_fp = '/home/ein/jinlong/std_data/ICDAR2019-LSTV/train_full_labels.json'
    label_dir = '/home/ein/jinlong/std_data/ICDAR2019-LSTV/train_gts'
    img_prefix_dir = 'ICDAR2019-LSTV/train_images'
    label_prefix_dir = 'ICDAR2019-LSTV/train_gts'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    idx_pairs = generate_icpr_2019_lstv_idx_pairs(
        label_json_fp, label_dir, img_prefix_dir, label_prefix_dir
    )
    save_idx_file(idx_pairs, '/home/ein/jinlong/std_data/ICDAR2019-LSTV/train.tsv')


if __name__ == '__main__':
    # icpr_mtwi_2018()
    # icdar_rctw_2017()
    icdar_2019_lstv()
