"""Evaluation: predictions, group metrics, figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config
from src.utils import save_json


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = logits.softmax(1)
        preds = probs.argmax(1)
        confs = probs.max(1).values
        for i in range(images.size(0)):
            y = int(batch["label"][i])
            p = int(preds[i])
            rows.append(
                {
                    "img_id": int(batch["img_id"][i]),
                    "true_label": y,
                    "background_label": int(batch["place"][i]),
                    "group_id": int(batch["group"][i]),
                    "prediction": p,
                    "confidence": float(confs[i]),
                    "prob_waterbird": float(probs[i, 1]),
                    "correct": int(p == y),
                }
            )
    return pd.DataFrame(rows)


def compute_metrics(df):
    y_true = df["true_label"].values
    y_pred = df["prediction"].values
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    group_acc = (
        df.groupby("group_id")["correct"].mean().round(4).to_dict()
    )
    group_counts = df.groupby("group_id").size().to_dict()
    metrics = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "f1_macro": float(macro_f1),
        "f1_landbird": float(f1[0]),
        "f1_waterbird": float(f1[1]),
        "precision_landbird": float(precision[0]),
        "recall_landbird": float(recall[0]),
        "precision_waterbird": float(precision[1]),
        "recall_waterbird": float(recall[1]),
        "support_landbird": int(support[0]),
        "support_waterbird": int(support[1]),
        "group_accuracy": {str(k): float(v) for k, v in group_acc.items()},
        "group_count": {str(k): int(v) for k, v in group_counts.items()},
        "worst_group_accuracy": float(min(group_acc.values())),
        "worst_group": str(min(group_acc, key=group_acc.get)),
        "minority_mean_accuracy": float(
            np.mean([group_acc.get(g, np.nan) for g in config.MINORITY_GROUPS])
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics


def plot_group_accuracy(metrics, out_path, title="Accuracy by group"):
    groups = sorted(int(g) for g in metrics["group_accuracy"])
    accs = [metrics["group_accuracy"][str(g)] for g in groups]
    labels = [config.GROUP_NAMES[g] for g in groups]
    colors = ["#4C72B0" if g in config.MINORITY_GROUPS else "#8C8C8C" for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(groups)), accs, color=colors)
    ax.axhline(metrics["accuracy"], ls="--", color="black", lw=1,
               label=f"overall {metrics['accuracy']:.3f}")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(metrics, out_path, title="Confusion matrix"):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_group_distribution(df, out_path, title="Test set group distribution"):
    counts = df.groupby("group_id").size()
    groups = sorted(counts.index)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["#4C72B0" if g in config.MINORITY_GROUPS else "#8C8C8C" for g in groups]
    bars = ax.bar(range(len(groups)), counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([config.GROUP_NAMES[g] for g in groups], fontsize=8)
    ax.set_ylabel("Images")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_and_save(model, loader, device, out_dir, tag):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = collect_predictions(model, loader, device)
    metrics = compute_metrics(preds)
    metrics["tag"] = tag

    preds.to_csv(out_dir / "per_image_predictions.csv", index=False)
    save_json(metrics, out_dir / "metrics.json")

    plot_group_accuracy(metrics, out_dir / "group_accuracy.png",
                        title=f"{tag}: accuracy by group")
    plot_confusion_matrix(metrics, out_dir / "confusion_matrix.png",
                          title=f"{tag}: confusion matrix")
    plot_group_distribution(preds, out_dir / "group_distribution.png",
                            title=f"{tag}: test group distribution")
    return preds, metrics
