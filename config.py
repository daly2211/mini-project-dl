"""Central configuration for the XAI spurious-correlation experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Paths
DATA_ROOT = ROOT / "data" / "raw"
METADATA_CSV = DATA_ROOT / "metadata.csv"
CUB_BBOX_DIR = DATA_ROOT / "cub_annotations"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"

for _d in (CHECKPOINT_DIR, RESULTS_DIR, FIGURE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42

# Data
IMG_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["landbird", "waterbird"]
PLACE_NAMES = ["land", "water"]
GROUP_NAMES = {
    0: "landbird / land (majority)",
    1: "landbird / water (minority)",
    2: "waterbird / land (minority)",
    3: "waterbird / water (majority)",
}
MINORITY_GROUPS = [1, 2]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Training
BATCH_SIZE = 32
NUM_WORKERS = 0  # single-process loading: lower host RAM, stable on Windows
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 3
USE_AMP = False  # GTX 1660 has no tensor cores; fp16 overflow corrupted BN stats

# XAI / attribution
BACKGROUND_RATIO_QUANTILE = 0.75  # top 25% background-reliant training images
TARGET_LAYER_NAME = "layer4.1"

# ImageNet pretrained weights (torchvision enum-free string, works across versions)
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
