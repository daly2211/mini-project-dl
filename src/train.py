"""Training loop with AMP and early stopping."""

import time

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import config
from src.model import build_model, get_device
from src.utils import save_json, set_seed


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_loss_acc(model, loader, criterion, device):
    """Validation loss/accuracy plus worst-group accuracy."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    group_correct, group_total = {}, {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)

        loss_sum += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for g, ok in zip(batch["group"].tolist(), (preds == labels).tolist()):
            group_correct[g] = group_correct.get(g, 0) + int(ok)
            group_total[g] = group_total.get(g, 0) + 1
    group_acc = {g: group_correct[g] / group_total[g] for g in sorted(group_total)}
    worst = min(group_acc.values())
    return loss_sum / total, correct / total, group_acc, worst


def run_training(tag, train_loader, val_loader, epochs=None, lr=None,
                 weight_decay=None, device=None, verbose=True):
    set_seed(config.SEED)
    device = device or get_device()
    amp = config.USE_AMP and device.type == "cuda"
    epochs = epochs or config.EPOCHS
    lr = lr or config.LR
    weight_decay = config.WEIGHT_DECAY if weight_decay is None else weight_decay

    model = build_model(pretrained=True).to(device)
    (config.RESULTS_DIR / tag).mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None

    history = []
    best_acc, best_epoch, best_state = -1.0, -1, None
    patience_left = config.EARLY_STOP_PATIENCE

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp
        )
        val_loss, val_acc, group_acc, worst = eval_loss_acc(
            model, val_loader, criterion, device
        )
        if not (np.isfinite(val_loss) and np.isfinite(val_acc)):
            raise RuntimeError(
                f"[{tag}] non-finite validation loss/accuracy at epoch {epoch}; "
                "check BatchNorm running statistics"
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_worst_group_acc": worst,
                **{f"val_group{g}_acc": a for g, a in group_acc.items()},
            }
        )
        if verbose:
            print(
                f"[{tag}] epoch {epoch}/{epochs} | train loss {train_loss:.4f} "
                f"acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} "
                f"worst-group {worst:.4f} | {time.time() - t0:.1f}s",
                flush=True,
            )
        pd.DataFrame(history).to_csv(
            config.RESULTS_DIR / tag / "history.csv", index=False
        )
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[{tag}] early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    model.to(device).eval()
    ckpt_path = config.CHECKPOINT_DIR / f"{tag}.pth"
    torch.save(
        {
            "state_dict": best_state,
            "epoch": best_epoch,
            "val_acc": best_acc,
            "config": {
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": train_loader.batch_size,
                "seed": config.SEED,
                "amp": amp,
            },
        },
        ckpt_path,
    )
    pd.DataFrame(history).to_csv(config.RESULTS_DIR / tag / "history.csv", index=False)
    save_json(
        {
            "tag": tag,
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "batch_size": train_loader.batch_size,
            "amp": amp,
            "device": str(device),
            "history": history,
        },
        config.RESULTS_DIR / tag / "training_log.json",
    )
    if verbose:
        print(f"[{tag}] best val acc {best_acc:.4f} @ epoch {best_epoch} -> {ckpt_path}", flush=True)
    return model, pd.DataFrame(history), best_acc
