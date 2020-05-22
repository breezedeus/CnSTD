import os
import glob


def generate_idx_pairs(img_dir, img_prefix_dir, label_prefix_dir, label_prefix_fn=''):
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


def save_idx_file(idx_pairs, output_fp):
    with open(output_fp, 'w') as f:
        for one_pair in idx_pairs:
            f.write('\t'.join(one_pair) + '\n')


def main():
    img_dir = 'data/icdar2015/train/images'
    img_prefix_dir = 'icdar2015/train/images'
    label_prefix_dir = 'icdar2015/train/gts'
    label_prefix_fn = 'gt_'
    idx_pairs = generate_idx_pairs(img_dir, img_prefix_dir, label_prefix_dir, label_prefix_fn)
    save_idx_file(idx_pairs, 'data/icdar2015/train.txt')


if __name__ == '__main__':
    main()