# coding: utf-8

from cnstd.utils.utils import sort_boxes


def four_to_eight(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_sort_boxes():
    # 一些用于测试的box坐标
    boxes = [
        [0, 2, 20, 18],
        [21, 1, 40, 19],
        [0, 20, 20, 40],
        [21, 20, 40, 40],
    ]
    boxes = [{'box': four_to_eight(box)} for box in boxes]
    out = sort_boxes(boxes, key='box')
    print(out)
