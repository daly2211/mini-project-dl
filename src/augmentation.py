"""Transforms and XAI-guided sample selection.

Baseline training uses mild augmentation. The proposed method applies stronger
background-oriented augmentation (aggressive crop, colour jitter, random
erasing) only to the training images whose baseline Grad-CAM attribution was
most concentrated outside the annotated bird box.
"""

import numpy as np
from torchvision import transforms

import config


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def get_train_transform(strong: bool = False):
    if strong:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    config.IMG_SIZE, scale=(0.45, 1.0), ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15
                ),
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
                transforms.RandomErasing(
                    p=0.35, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def select_background_dependent(df, quantile=None, column="background_attribution_ratio"):
    """Return training indices above a ratio threshold computed on TRAIN only.

    `df` must be the train-split per-image attribution table (indexed 0..N-1,
    matching the train dataset order). The threshold never touches val/test.
    """
    quantile = config.BACKGROUND_RATIO_QUANTILE if quantile is None else quantile
    threshold = float(np.nanquantile(df[column].values, quantile))
    indices = df.index[df[column] >= threshold].tolist()
    return indices, threshold
