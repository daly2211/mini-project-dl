"""End-to-end experiment runner.

Usage:
    python run_experiments.py --steps baseline,gradcam,rq2,xai,compare
    python run_experiments.py            # runs everything

Steps:
    baseline : train (or reuse) and evaluate the ERM ResNet18
    gradcam  : Grad-CAM attribution pass + example grid (baseline, test)
    rq2      : association between background reliance and errors (baseline)
    xai      : select background-dependent train images, train + evaluate
               the XAI-guided augmentation model
    compare  : baseline-vs-modified table and figure
"""

import argparse
import shutil

import pandas as pd
import torch

import config
from src.analysis import analyze_rq2
from src.augmentation import select_background_dependent
from src.compare import build_comparison
from src.dataset import build_eval_loader, build_train_loader, load_metadata
from src.evaluate import evaluate_and_save
from src.gradcam import run_attribution_pass, visualize_examples
from src.model import describe_device, get_device, load_checkpoint
from src.train import run_training
from src.utils import save_json, set_seed


def copy_to_figures(src, dst_name):
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, config.FIGURE_DIR / dst_name)


def train_or_load_baseline(metadata, device):
    ckpt = config.CHECKPOINT_DIR / "baseline.pth"
    if ckpt.exists():
        print(f"[baseline] reusing checkpoint {ckpt}", flush=True)
        model, _ = load_checkpoint(ckpt, device)
        return model
    print("[baseline] training from scratch", flush=True)
    train_loader = build_train_loader(df=metadata)
    val_loader = build_eval_loader("val", df=metadata)
    model, _, _ = run_training("baseline", train_loader, val_loader, device=device)
    return model


def step_baseline(metadata, device):
    model = train_or_load_baseline(metadata, device)
    test_loader = build_eval_loader("test", df=metadata)
    evaluate_and_save(model, test_loader, device, config.RESULTS_DIR / "baseline",
                      tag="baseline")
    copy_to_figures(config.RESULTS_DIR / "baseline" / "group_accuracy.png",
                    "group_accuracy.png")
    copy_to_figures(config.RESULTS_DIR / "baseline" / "confusion_matrix.png",
                    "confusion_matrix.png")
    return model


def step_gradcam(metadata, device, tag="baseline"):
    model, _ = load_checkpoint(config.CHECKPOINT_DIR / f"{tag}.pth", device)
    test_loader = build_eval_loader("test", df=metadata)
    out_dir = config.RESULTS_DIR / tag
    per_image = run_attribution_pass(
        model, test_loader, device,
        out_dir / "per_image_attributions.csv", split="test",
    )
    visualize_examples(
        model, metadata, per_image, out_dir / "gradcam_examples.png", device,
        tag=tag,
    )
    if tag == "baseline":
        copy_to_figures(out_dir / "gradcam_examples.png", "gradcam_examples.png")
    print(f"[gradcam:{tag}] wrote {out_dir / 'per_image_attributions.csv'}", flush=True)
    return per_image


def step_rq2(metadata, device, tag="baseline"):
    out_dir = config.RESULTS_DIR / tag
    per_image = pd.read_csv(out_dir / "per_image_attributions.csv")
    summary = analyze_rq2(per_image, out_dir / "rq2", tag=tag)
    if tag == "baseline":
        copy_to_figures(out_dir / "rq2" / "background_reliance_vs_correctness.png",
                        "background_reliance.png")
        copy_to_figures(out_dir / "rq2" / "background_reliance_by_group.png",
                        "attribution_by_group.png")
        copy_to_figures(out_dir / "rq2" / "attribution_vs_confidence.png",
                        "attribution_vs_confidence.png")
    return summary


def step_xai(metadata, device):
    # 1. Attribute TRAIN images with the baseline (no test data involved).
    baseline_model, _ = load_checkpoint(config.CHECKPOINT_DIR / "baseline.pth", device)
    train_loader = build_train_loader(df=metadata, shuffle=False)
    train_attr = run_attribution_pass(
        baseline_model, train_loader, device,
        config.RESULTS_DIR / "baseline" / "train_attributions.csv", split="train",
    )
    indices, threshold = select_background_dependent(train_attr)
    save_json(
        {
            "threshold": threshold,
            "quantile": config.BACKGROUND_RATIO_QUANTILE,
            "n_selected": len(indices),
            "n_train": int(len(train_attr)),
        },
        config.RESULTS_DIR / "baseline" / "selection.json",
    )
    print(f"[xai] selected {len(indices)}/{len(train_attr)} train images "
          f"(ratio >= {threshold:.4f})", flush=True)

    # 2. Train the same architecture with stronger augmentation on those images.
    torch.cuda.empty_cache()
    xai_train_loader = build_train_loader(df=metadata, strong_indices=indices)
    val_loader = build_eval_loader("val", df=metadata)
    model, _, _ = run_training("xai_guided", xai_train_loader, val_loader,
                               device=device)

    # 3. Evaluate identically.
    test_loader = build_eval_loader("test", df=metadata)
    evaluate_and_save(model, test_loader, device,
                      config.RESULTS_DIR / "xai_guided", tag="xai_guided")
    return model


def step_compare():
    import json

    def read_json(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    base_metrics = read_json(config.RESULTS_DIR / "baseline" / "metrics.json")
    xai_metrics = read_json(config.RESULTS_DIR / "xai_guided" / "metrics.json")

    def read_rq2(tag):
        p = config.RESULTS_DIR / tag / "rq2" / "rq2_analysis.json"
        return read_json(p) if p.exists() else None

    table = build_comparison(
        base_metrics, xai_metrics, read_rq2("baseline"), read_rq2("xai_guided"),
        config.RESULTS_DIR, config.FIGURE_DIR,
    )
    print(table.to_string(index=False), flush=True)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", default="baseline,gradcam,rq2,xai,compare",
        help="comma-separated subset of baseline,gradcam,rq2,xai,compare",
    )
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    set_seed(config.SEED)
    device = get_device()
    print(describe_device(), flush=True)

    metadata = load_metadata()
    print(
        metadata.groupby(["split", "group"]).size().unstack(fill_value=0).to_string(),
        flush=True,
    )

    for step in steps:
        print(f"\n===== {step} =====", flush=True)
        if step == "baseline":
            step_baseline(metadata, device)
        elif step == "gradcam":
            step_gradcam(metadata, device, tag="baseline")
        elif step == "rq2":
            step_rq2(metadata, device, tag="baseline")
        elif step == "xai":
            step_xai(metadata, device)
            step_gradcam(metadata, device, tag="xai_guided")
            step_rq2(metadata, device, tag="xai_guided")
        elif step == "compare":
            step_compare()
        else:
            raise ValueError(f"unknown step {step}")


if __name__ == "__main__":
    main()
