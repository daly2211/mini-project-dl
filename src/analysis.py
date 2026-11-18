"""RQ2 analysis: association between background reliance and errors."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from src.utils import save_json


def _boxplot(ax, data, labels, colors, ylabel):
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops={"color": "black"})
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel(ylabel)


def analyze_rq2(df, out_dir, tag="baseline"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=["background_attribution_ratio"]).copy()

    correct = df[df.correct == 1]["background_attribution_ratio"]
    incorrect = df[df.correct == 0]["background_attribution_ratio"]

    summary = {
        "tag": tag,
        "n_images": int(len(df)),
        "n_correct": int(len(correct)),
        "n_incorrect": int(len(incorrect)),
        "ratio_mean_all": float(df["background_attribution_ratio"].mean()),
        "ratio_median_all": float(df["background_attribution_ratio"].median()),
        "ratio_mean_correct": float(correct.mean()),
        "ratio_median_correct": float(correct.median()),
        "ratio_mean_incorrect": float(incorrect.mean()),
        "ratio_median_incorrect": float(incorrect.median()),
    }

    if len(correct) > 0 and len(incorrect) > 0:
        u = stats.mannwhitneyu(correct, incorrect, alternative="two-sided")
        summary["mannwhitney_u"] = float(u.statistic)
        summary["mannwhitney_p"] = float(u.pvalue)
        rho, p = stats.spearmanr(
            df["background_attribution_ratio"], df["correct"]
        )
        summary["spearman_r_ratio_vs_correct"] = float(rho)
        summary["spearman_p"] = float(p)

        # Logistic regression: P(error) ~ standardized ratio (reported as association)
        x = StandardScaler().fit_transform(
            df[["background_attribution_ratio"]].values
        )
        y = 1 - df["correct"].values
        if len(np.unique(y)) == 2:
            clf = LogisticRegression().fit(x, y)
            summary["logit_coef_standardized"] = float(clf.coef_[0, 0])
            summary["logit_intercept"] = float(clf.intercept_[0])
            summary["auc_ratio_predicts_error"] = float(
                roc_auc_score(y, df["background_attribution_ratio"])
            )

    group_stats = {}
    for g in sorted(df.group_id.unique()):
        vals = df[df.group_id == g]["background_attribution_ratio"]
        group_stats[str(int(g))] = {
            "name": config.GROUP_NAMES[int(g)],
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "accuracy": float(df[df.group_id == g]["correct"].mean()),
        }
    summary["group_stats"] = group_stats
    minority_means = [group_stats[str(g)]["mean"] for g in config.MINORITY_GROUPS
                      if str(g) in group_stats]
    majority_means = [group_stats[str(g)]["mean"] for g in (0, 3)
                      if str(g) in group_stats]
    summary["ratio_mean_minority_groups"] = float(np.mean(minority_means))
    summary["ratio_mean_majority_groups"] = float(np.mean(majority_means))

    save_json(summary, out_dir / "rq2_analysis.json")

    # Graph 1: background reliance vs correctness
    fig, ax = plt.subplots(figsize=(5.2, 4))
    _boxplot(
        ax,
        [correct.values, incorrect.values],
        [f"correct (n={len(correct)})", f"incorrect (n={len(incorrect)})"],
        ["#4C72B0", "#C44E52"],
        "Background attribution ratio",
    )
    ax.set_title(f"{tag}: background reliance vs. correctness")
    fig.tight_layout()
    fig.savefig(out_dir / "background_reliance_vs_correctness.png", dpi=150)
    plt.close(fig)

    # Graph 2: background reliance across the four groups
    fig, ax = plt.subplots(figsize=(8, 4))
    data, labels, colors = [], [], []
    for g in sorted(df.group_id.unique()):
        data.append(df[df.group_id == g]["background_attribution_ratio"].values)
        labels.append(config.GROUP_NAMES[int(g)].replace(" (", "\n("))
        colors.append("#4C72B0" if int(g) in config.MINORITY_GROUPS else "#8C8C8C")
    _boxplot(ax, data, labels, colors, "Background attribution ratio")
    ax.set_title(f"{tag}: background reliance by group")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "background_reliance_by_group.png", dpi=150)
    plt.close(fig)

    # Graph 3: attribution ratio vs confidence
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ok = df[df.correct == 1]
    bad = df[df.correct == 0]
    ax.scatter(ok["background_attribution_ratio"], ok["confidence"], s=6,
               alpha=0.35, label="correct", color="#4C72B0")
    ax.scatter(bad["background_attribution_ratio"], bad["confidence"], s=14,
               alpha=0.75, label="incorrect", color="#C44E52", marker="x")
    ax.set_xlabel("Background attribution ratio")
    ax.set_ylabel("Prediction confidence")
    ax.set_title(f"{tag}: attribution ratio vs. confidence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "attribution_vs_confidence.png", dpi=150)
    plt.close(fig)

    return summary
