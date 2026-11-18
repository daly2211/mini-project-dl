"""Model construction and device helpers."""

import torch
from torchvision import models

import config


def build_model(pretrained: bool = True, num_classes: int = config.NUM_CLASSES):
    weights = config.PRETRAINED_WEIGHTS if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device():
    if not torch.cuda.is_available():
        return "CUDA unavailable -> CPU"
    dev = torch.cuda.get_device_properties(0)
    return (
        f"GPU: {dev.name} | VRAM: {dev.total_memory / 1024**3:.1f} GiB | "
        f"CUDA {torch.version.cuda}"
    )


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
