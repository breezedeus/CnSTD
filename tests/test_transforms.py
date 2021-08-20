from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

from cnstd.utils import (
    set_logger,
    data_dir,
    load_model_params,
    imsave,
    imread,
)

EXAMPLE_DIR = Path(__file__).parent.parent / 'examples'


def test_transforms():
    train_transform = T.Compose(  # MUST NOT include `Resize`
        [
            # T.RandomInvert(p=1.0),
            T.RandomPosterize(bits=4, p=1.0),
            T.RandomAdjustSharpness(sharpness_factor=0.5, p=1.0),
            # T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2),
            # T.RandomEqualize(p=0.3),
            # T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
        ]
    )
    img_fp = EXAMPLE_DIR / '1_res.png'
    img = imread(img_fp)
    img = train_transform(torch.from_numpy(img))
    imsave(img.numpy().transpose((1, 2, 0)), 'test-transformed.png', normalized=False)
