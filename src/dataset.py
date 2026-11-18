"""Waterbirds dataset loading.

The official WILDS metadata.csv contains one row per image:
    img_id, img_filename, y (bird label), split (0/1/2), place (background),
    place_filename.

WILDS ships no bounding boxes, so CUB-200-2011 boxes (same images and
filenames) are joined in from `data/raw/cub_annotations/`.

Group definition follows Sagawa et al. (2020): group = 2 * y + place.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import config
from src.augmentation import get_eval_transform, get_train_transform

BBOX_COLS = ["x_1", "y_1", "x_2", "y_2"]
SPLIT_MAP = {0: "train", 1: "val", 2: "test"}


def _attach_cub_bboxes(df, bbox_dir=None):
    bbox_dir = Path(bbox_dir or config.CUB_BBOX_DIR)
    images = pd.read_csv(
        bbox_dir / "images.txt", sep=r"\s+", header=None,
        names=["cub_id", "img_filename"],
    )
    boxes = pd.read_csv(
        bbox_dir / "bounding_boxes.txt", sep=r"\s+", header=None,
        names=["cub_id", "bx", "by", "bw", "bh"],
    )
    cub = images.merge(boxes, on="cub_id")[["img_filename", "bx", "by", "bw", "bh"]]
    cub = cub.assign(
        x_1=cub["bx"], y_1=cub["by"],
        x_2=cub["bx"] + cub["bw"], y_2=cub["by"] + cub["bh"],
    )[["img_filename", "x_1", "y_1", "x_2", "y_2"]]
    merged = df.merge(cub, on="img_filename", how="left")
    missing = int(merged["x_1"].isna().sum())
    if missing:
        raise ValueError(f"{missing} images have no CUB bounding box")
    return merged


def load_metadata(path=None):
    path = Path(path or config.METADATA_CSV)
    df = pd.read_csv(path)
    if df["split"].dtype != object:
        df["split"] = df["split"].map(SPLIT_MAP)
    if not set(BBOX_COLS).issubset(df.columns):
        df = _attach_cub_bboxes(df)
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    df["group"] = 2 * df["y"] + df["place"]
    return df


class WaterbirdsDataset(Dataset):
    """Returns image, bird label, background label, group id, id, bbox.

    Bounding boxes are returned in normalized [0, 1] coordinates relative to the
    original image so they can be applied to any resized image.
    """

    def __init__(self, split, df=None, base_transform=None, strong_indices=None,
                 strong_transform=None, root=None):
        self.root = Path(root or config.DATA_ROOT)
        df = load_metadata() if df is None else df
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.base_transform = base_transform
        self.strong_transform = strong_transform
        self.strong_indices = set(strong_indices or [])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.root / row["img_filename"]).convert("RGB")
        w, h = image.size

        tf = (
            self.strong_transform
            if (self.strong_transform is not None and idx in self.strong_indices)
            else self.base_transform
        )
        tensor = tf(image)

        bbox = torch.tensor(
            [
                row["x_1"] / w,
                row["y_1"] / h,
                row["x_2"] / w,
                row["y_2"] / h,
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        return {
            "image": tensor,
            "label": int(row["y"]),
            "place": int(row["place"]),
            "group": int(row["group"]),
            "img_id": int(row["img_id"]),
            "bbox": bbox,
        }


def build_train_loader(df=None, strong_indices=None, batch_size=None, shuffle=True):
    ds = WaterbirdsDataset(
        "train",
        df=df,
        base_transform=get_train_transform(strong=False),
        strong_indices=strong_indices,
        strong_transform=get_train_transform(strong=True) if strong_indices else None,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.NUM_WORKERS > 0,
        drop_last=False,
        persistent_workers=False,
    )


def build_eval_loader(split, df=None, batch_size=None):
    ds = WaterbirdsDataset(split, df=df, base_transform=get_eval_transform())
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.NUM_WORKERS > 0,
    )
