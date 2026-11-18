"""Generate presentation-ready assets from the experiment outputs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from src.dataset import load_metadata
from src.gradcam import _load_image_for_display, _single_cam, make_cam
from src.model import get_device, load_checkpoint

OUT = Path(__file__).resolve().parents[1] / "slides_assets"
OUT.mkdir(exist_ok=True)

INK = "#12131A"
MUTED = "#6B7280"
BLUE = "#2F6FED"
RED = "#D6453D"
GREEN = "#1E9E6A"
GRAY = "#9AA0AA"
LIGHT = "#E7E9EE"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#C9CDD4",
        "axes.linewidth": 1.2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

device = get_device()
model, _ = load_checkpoint(config.CHECKPOINT_DIR / "baseline.pth", device)
meta = load_metadata().set_index("img_id")
attr = pd.read_csv(config.RESULTS_DIR / "baseline" / "per_image_attributions.csv")
test = attr[attr.split == "test"]


def rgb_for(img_id):
    path = config.DATA_ROOT / meta.loc[img_id, "img_filename"]
    return _load_image_for_display(path)


def cam_for(img_id):
    from src.augmentation import get_eval_transform

    rgb = rgb_for(img_id)
    t = get_eval_transform()(Image.fromarray((rgb * 255).astype(np.uint8)))
    gray = _single_cam(model, t.unsqueeze(0), device)
    return rgb, gray


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- slide 1 photos
# Chosen from a contact sheet of high-confidence correct test images whose
# background source (place_filename root) matches the annotation.
GROUP_IMAGES = {0: 11253, 1: 9805, 2: 3061, 3: 3066}

for g, img_id in GROUP_IMAGES.items():
    rgb = rgb_for(img_id)
    fig = plt.figure(figsize=(4.2, 3.55), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb)
    ax.axis("off")
    save(fig, f"photo_group{g}.png")

# ------------------------------------------------------- slide 4 example overlays
ok = test[(test.correct == 1) & (test.confidence > 0.9)].sort_values(
    "background_attribution_ratio"
).iloc[0]
bad = test[(test.correct == 0)].sort_values("background_attribution_ratio").iloc[-1]

for tag, row in (("correct", ok), ("incorrect", bad)):
    rgb, gray = cam_for(row.img_id)
    overlay = show_cam_on_image(rgb, gray, use_rgb=True) / 255.0
    fig = plt.figure(figsize=(4.4, 3.7), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(overlay)
    ax.axis("off")
    save(fig, f"gradcam_{tag}.png")
    print(
        f"  {tag}: img_id={int(row.img_id)} ratio={row.background_attribution_ratio:.3f} "
        f"conf={row.confidence:.3f} true={int(row.true_label)} pred={int(row.prediction)}"
    )

# ------------------------------------------------------------ bbox explainer
bbox_row = test[test.img_id == 3061].iloc[0]
rgb = rgb_for(bbox_row.img_id)
bx = meta.loc[bbox_row.img_id, ["x_1", "y_1", "x_2", "y_2"]].astype(float).values
orig = Image.open(config.DATA_ROOT / meta.loc[bbox_row.img_id, "img_filename"])
w, h = orig.size
norm = np.array([bx[0] / w, bx[1] / h, bx[2] / w, bx[3] / h])

fig = plt.figure(figsize=(5.6, 3.7), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(rgb)
ax.axis("off")
H, W = rgb.shape[:2]
x1, y1, x2, y2 = norm[0] * W, norm[1] * H, norm[2] * W, norm[3] * H
ax.add_patch(
    Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2.2, edgecolor="#00E5A0")
)
ax.text(
    (x1 + x2) / 2, y1 - 6, "bird box", color="#00E5A0", fontsize=11,
    ha="center", va="bottom", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.65),
)
ax.annotate(
    "background = outside the box",
    xy=(W * 0.09, H * 0.12), xytext=(W * 0.05, H * 0.05),
    color="white", fontsize=10, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc=RED, ec="none", alpha=0.9),
)
save(fig, "bbox_explainer.png")

# ------------------------------------------------------------- slide 3 bar chart
with open(config.RESULTS_DIR / "baseline" / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)
accs = [metrics["group_accuracy"][str(g)] for g in (0, 1, 2, 3)]
labels = ["Landbird\n/ land", "Landbird\n/ water", "Waterbird\n/ land", "Waterbird\n/ water"]
colors = [GRAY, GRAY, RED, GRAY]

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
bars = ax.bar(range(4), accs, color=colors, width=0.62, zorder=3)
for b, v in zip(bars, accs):
    ax.text(
        b.get_x() + b.get_width() / 2, v + 0.025, f"{v*100:.1f}%",
        ha="center", fontsize=15, fontweight="bold",
        color=RED if v < 0.6 else INK,
    )
ax.axhline(metrics["accuracy"], ls=(0, (5, 4)), lw=1.8, color=BLUE, zorder=4)
ax.text(
    -0.42, metrics["accuracy"] + 0.03, f"overall {metrics['accuracy']*100:.2f}%",
    color=BLUE, fontsize=11, fontweight="bold", ha="left",
)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
ax.set_ylabel("Test accuracy", fontsize=12)
ax.grid(axis="y", color=LIGHT, lw=1, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "chart_group_accuracy.png")

# ------------------------------------------------------------- slide 4 bar chart
with open(config.RESULTS_DIR / "baseline" / "rq2" / "rq2_analysis.json", encoding="utf-8") as f:
    rq2 = json.load(f)
vals = [rq2["ratio_mean_correct"], rq2["ratio_mean_incorrect"]]
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
bars = ax.bar([0, 1], vals, color=[BLUE, RED], width=0.5, zorder=3)
for b, v in zip(bars, vals):
    ax.text(
        b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.2f}%",
        ha="center", fontsize=24, fontweight="bold",
    )
ax.set_xticks([0, 1])
ax.set_xticklabels(
    [f"Correct predictions\n(n={rq2['n_correct']})", f"Incorrect predictions\n(n={rq2['n_incorrect']})"],
    fontsize=13,
)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=11)
ax.set_ylabel("Grad-CAM activation outside bird box", fontsize=12)
ax.grid(axis="y", color=LIGHT, lw=1, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "chart_correct_vs_incorrect.png")

# --------------------------------------------------------------- workflow diagram
fig, ax = plt.subplots(figsize=(10.8, 3.6), dpi=200)
ax.set_xlim(0, 10.8)
ax.set_ylim(0, 3.6)
ax.axis("off")


def box(x, y, w, h, label, color, fontsize=13, text_color="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            fc=color, ec="none", zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
            lw=2.2, color=MUTED, zorder=1,
        )
    )


# Bird image (photo above label, inside a dark card)
ax.add_patch(
    FancyBboxPatch((0.15, 0.75), 2.05, 2.05, boxstyle="round,pad=0.06,rounding_size=0.12",
                   fc="#1F2937", ec="none", zorder=2)
)
ax.imshow(Image.open(OUT / "photo_group2.png"), extent=(0.33, 2.02, 1.42, 2.62),
          zorder=3, aspect="auto")
ax.text(1.18, 1.06, "Bird image", ha="center", va="center",
        fontsize=13, color="white", fontweight="bold", zorder=4)

arrow(2.28, 1.75, 2.98, 1.75)
box(3.03, 1.18, 2.3, 1.18, "ResNet18\n(ImageNet)", BLUE)
arrow(5.38, 1.75, 6.08, 1.75)
box(6.13, 2.02, 2.05, 0.88, "Prediction", GREEN, fontsize=12)
box(6.13, 0.68, 2.05, 0.88, "Grad-CAM\nheatmap", RED, fontsize=12)

arrow(8.25, 3.05, 9.15, 3.05)
ax.imshow(Image.open(OUT / "gradcam_incorrect.png"), extent=(9.2, 10.7, 1.5, 3.45),
          zorder=3, aspect="auto")
ax.add_patch(Rectangle((9.2, 1.5), 1.5, 1.95, fill=False, lw=1.2,
                       edgecolor="#C9CDD4", zorder=4))
ax.text(9.95, 1.22, "attention map", ha="center", va="center",
        fontsize=11.5, color=MUTED, fontweight="bold")

save(fig, "workflow.png")

print("done")
