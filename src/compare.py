"""Baseline vs XAI-guided comparison table and figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.utils import save_json


def _ratio_summary(rq2):
    if not rq2:
        return {}
    return {
        "mean_bg_ratio_all": rq2.get("ratio_mean_all", np.nan),
        "mean_bg_ratio_correct": rq2.get("ratio_mean_correct", np.nan),
        "mean_bg_ratio_incorrect": rq2.get("ratio_mean_incorrect", np.nan),
        "ratio_minority_groups": rq2.get("ratio_mean_minority_groups", np.nan),
        "ratio_majority_groups": rq2.get("ratio_mean_majority_groups", np.nan),
    }


def build_comparison(baseline_metrics, xai_metrics, baseline_rq2, xai_rq2,
                     out_dir, figure_dir):
    out_dir = Path(out_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    def row(name, b, x):
        return {"metric": name, "baseline": b, "xai_guided": x}

    rows = [
        row("overall_accuracy", baseline_metrics["accuracy"], xai_metrics["accuracy"]),
        row("worst_group_accuracy", baseline_metrics["worst_group_accuracy"],
            xai_metrics["worst_group_accuracy"]),
        row("minority_mean_accuracy", baseline_metrics["minority_mean_accuracy"],
            xai_metrics["minority_mean_accuracy"]),
        row("group0_landbird_land", baseline_metrics["group_accuracy"]["0"],
            xai_metrics["group_accuracy"]["0"]),
        row("group1_landbird_water", baseline_metrics["group_accuracy"]["1"],
            xai_metrics["group_accuracy"]["1"]),
        row("group2_waterbird_land", baseline_metrics["group_accuracy"]["2"],
            xai_metrics["group_accuracy"]["2"]),
        row("group3_waterbird_water", baseline_metrics["group_accuracy"]["3"],
            xai_metrics["group_accuracy"]["3"]),
        row("f1_macro", baseline_metrics["f1_macro"], xai_metrics["f1_macro"]),
    ]

    br, xr = _ratio_summary(baseline_rq2), _ratio_summary(xai_rq2)
    for key in ["mean_bg_ratio_all", "mean_bg_ratio_correct",
                "mean_bg_ratio_incorrect", "ratio_minority_groups",
                "ratio_majority_groups"]:
        if key in br and key in xr:
            rows.append(row(key, br[key], xr[key]))

    table = pd.DataFrame(rows)
    delta = table["xai_guided"] - table["baseline"]
    # For accuracy, higher is better; for bg ratio, lower is better.
    table["delta"] = delta
    table.to_csv(out_dir / "metrics_comparison.csv", index=False)
    save_json(table.to_dict(orient="records"), out_dir / "metrics_comparison.json")

    # Figure: group accuracy comparison
    groups = [0, 1, 2, 3]
    labels = ["overall"] + [config.GROUP_NAMES[g].replace(" (", "\n(") for g in groups]
    b_vals = [baseline_metrics["accuracy"]] + [
        baseline_metrics["group_accuracy"][str(g)] for g in groups
    ]
    x_vals = [xai_metrics["accuracy"]] + [
        xai_metrics["group_accuracy"][str(g)] for g in groups
    ]
    idx = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.bar(idx - width / 2, b_vals, width, label="baseline", color="#8C8C8C")
    ax.bar(idx + width / 2, x_vals, width, label="XAI-guided", color="#4C72B0")
    for i, (b, x) in enumerate(zip(b_vals, x_vals)):
        ax.text(i - width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
        ax.text(i + width / 2, x + 0.01, f"{x:.2f}", ha="center", fontsize=7)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: baseline vs XAI-guided")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    if br and xr:
        keys = [
            ("all images", "mean_bg_ratio_all"),
            ("correct", "mean_bg_ratio_correct"),
            ("incorrect", "mean_bg_ratio_incorrect"),
            ("minority groups", "ratio_minority_groups"),
            ("majority groups", "ratio_majority_groups"),
        ]
        klabels = [k for k, kk in keys]
        bvals = [br[kk] for _, kk in keys]
        xvals = [xr[kk] for _, kk in keys]
        idx2 = np.arange(len(klabels))
        ax.bar(idx2 - width / 2, bvals, width, label="baseline", color="#8C8C8C")
        ax.bar(idx2 + width / 2, xvals, width, label="XAI-guided", color="#4C72B0")
        for i, (b, x) in enumerate(zip(bvals, xvals)):
            ax.text(i - width / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=7)
            ax.text(i + width / 2, x + 0.005, f"{x:.2f}", ha="center", fontsize=7)
        ax.set_xticks(idx2)
        ax.set_xticklabels(klabels, fontsize=8)
        ax.set_ylabel("Mean background attribution ratio")
        ax.set_title("Background reliance: baseline vs XAI-guided")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Attribution metrics unavailable", ha="center")

    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_vs_modified.png", dpi=150)
    plt.close(fig)
    return table
