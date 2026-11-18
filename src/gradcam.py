"""Grad-CAM analysis and background-attribution metrics.

Grad-CAM reference: Selvaraju et al. (2017).
Background attribution ratio = sum(CAM outside bird box) / sum(CAM).

Limitation: the Waterbirds annotation is a bounding box, not a segmentation
mask. Attribution outside the box can still fall on parts of the bird (wings,
tail) and inside the box can still be background. The ratio is therefore an
*approximation* of background reliance, and is reported as such.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

import config
from src.augmentation import get_eval_transform

SOURCE_TO_MODEL_SIZE = config.IMG_SIZE


def make_cam(model):
    return GradCAM(model=model, target_layers=[model.layer4[-1]])


def background_attribution_ratio(cam, bbox_norm):
    """Fraction of Grad-CAM mass outside the normalized bird bounding box."""
    h, w = cam.shape
    x1, y1, x2, y2 = np.asarray(bbox_norm, dtype=float) * np.array([w, h, w, h])
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 1, w))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 1, h))
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    total = float(cam.sum())
    if total <= 1e-12 or mask.all():
        return float("nan")
    return float(cam[~mask].sum() / total)


def run_attribution_pass(model, loader, device, out_csv, split):
    """Per-image predictions + Grad-CAM background-attribution ratio."""
    model.eval()
    cam = make_cam(model)
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.enable_grad():
            grays = cam(input_tensor=images, targets=None)
        with torch.no_grad():
            probs = model(images).softmax(1)
        preds = probs.argmax(1)
        confs = probs.max(1).values

        for i in range(images.size(0)):
            y = int(batch["label"][i])
            p = int(preds[i])
            bbox = batch["bbox"][i].numpy()
            rows.append(
                {
                    "img_id": int(batch["img_id"][i]),
                    "split": split,
                    "true_label": y,
                    "background_label": int(batch["place"][i]),
                    "group_id": int(batch["group"][i]),
                    "prediction": p,
                    "confidence": float(confs[i]),
                    "prob_waterbird": float(probs[i, 1]),
                    "correct": int(p == y),
                    "background_attribution_ratio": background_attribution_ratio(
                        np.asarray(grays[i], dtype=np.float64), bbox
                    ),
                }
            )
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def _load_image_for_display(path):
    img = Image.open(path).convert("RGB").resize(
        (SOURCE_TO_MODEL_SIZE, SOURCE_TO_MODEL_SIZE), Image.BILINEAR
    )
    return np.asarray(img, dtype=np.float32) / 255.0


def _single_cam(model, image_tensor, device):
    cam = make_cam(model)
    with torch.enable_grad():
        gray = cam(input_tensor=image_tensor.to(device), targets=None)[0]
    return np.asarray(gray)


def visualize_examples(model, metadata_df, per_image_df, out_path, device,
                       n_cols=3, tag="baseline"):
    """Grid: one row per group, columns = highest / median / lowest ratio."""
    meta = metadata_df.set_index("img_id")
    rows = []
    for group in sorted(config.GROUP_NAMES):
        sub = per_image_df[per_image_df.group_id == group].sort_values(
            "background_attribution_ratio"
        )
        if sub.empty:
            continue
        picks = [
            sub.iloc[-1],  # highest background reliance
            sub.iloc[len(sub) // 2],  # median
            sub.iloc[0],  # lowest
        ]
        rows.append((group, picks[:n_cols]))

    eval_tf = get_eval_transform()
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(3.4 * n_cols, 2.9 * len(rows)),
        squeeze=False,
    )
    col_titles = ["highest bg ratio", "median bg ratio", "lowest bg ratio"]

    for r, (group, picks) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(picks):
                continue
            row = picks[c]
            img_path = config.DATA_ROOT / meta.loc[row.img_id, "img_filename"]
            rgb = _load_image_for_display(img_path)
            tensor = eval_tf(Image.fromarray((rgb * 255).astype(np.uint8)))
            gray = _single_cam(model, tensor.unsqueeze(0), device)
            overlay = show_cam_on_image(rgb, gray, use_rgb=True)
            ax.imshow(overlay)
            ax.set_title(
                f"{config.GROUP_NAMES[group]}\n"
                f"true={config.CLASS_NAMES[row.true_label]} "
                f"pred={config.CLASS_NAMES[row.prediction]} ({row.confidence:.2f})\n"
                f"bg ratio {row.background_attribution_ratio:.2f} "
                f"{'OK' if row.correct else 'WRONG'}",
                fontsize=7,
            )
            if r == 0:
                ax.text(
                    0.5, 1.35, col_titles[c], transform=ax.transAxes,
                    ha="center", fontsize=9, fontweight="bold",
                )

    fig.suptitle(f"Grad-CAM examples ({tag}, test set)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
