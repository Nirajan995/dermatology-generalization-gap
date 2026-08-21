"""Paths and shared constants.

Every path is overridable by an environment variable so the same code runs on
Kaggle (where the original experiments were done), on Colab, and locally.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Root directories ------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("DERMGAP_DATA", REPO_ROOT / "data"))
ARTIFACT_ROOT = Path(os.environ.get("DERMGAP_ARTIFACTS", REPO_ROOT / "artifacts"))

UNIFIED_DIR = ARTIFACT_ROOT / "unified"        # unified CSV tables
EMBED_DIR = ARTIFACT_ROOT / "embeddings"       # frozen features (.npy) + meta
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"  # cancer baseline weights
RESULTS_DIR = Path(os.environ.get("DERMGAP_RESULTS", REPO_ROOT / "results"))
FIGURE_DIR = Path(os.environ.get("DERMGAP_FIGURES", REPO_ROOT / "figures"))

# --- Raw dataset locations -------------------------------------------------
# Defaults follow the Kaggle mount layout used for the paper. Override with
# env vars (or edit) when running elsewhere. See docs/DATA.md.

DATASET_PATHS = {
    "ham10000": Path(os.environ.get(
        "DERMGAP_HAM10000", "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000")),
    "isic2019": Path(os.environ.get(
        "DERMGAP_ISIC2019", "/kaggle/input/datasets/andrewmvd/isic-2019")),
    "ddi": Path(os.environ.get(
        "DERMGAP_DDI", "/kaggle/input/datasets/nirajankunwor/ddi-dataset")),
    "scin": Path(os.environ.get(
        "DERMGAP_SCIN", "/kaggle/input/datasets/nirajankunwor/scin-dataset")),
}

# --- Experiment constants --------------------------------------------------

MODELS = ["resnet_baseline", "dermlip", "monet", "dinov3"]

MODEL_DISPLAY_NAMES = {
    "resnet_baseline": "Cancer\nbaseline",
    "dermlip": "DermLIP",
    "monet": "MONET",
    "dinov3": "DINOv3",
}

# Kept here rather than in models.py so that CPU-only analysis never imports torch.
EMBED_DIMS = {
    "resnet_baseline": 2048,
    "dermlip": 512,
    "monet": 1024,
    "dinov3": 768,
}

EVAL_DATASETS = ["ddi", "scin", "hamisic_test"]

# Column names differ per evaluation table.
LABEL_COL = {"ddi": "label", "scin": "label", "hamisic_test": "mapped_label"}
GROUP_COL = {"ddi": "patient_id", "scin": "patient_id", "hamisic_test": "pid"}

SEED = 42
N_BOOT = 1000
TEST_SIZE = 0.30

# Minimum images per class before a class enters a probe / purity computation.
MIN_CLASS_COUNT_PROBE = 10
MIN_CLASS_COUNT_PURITY = {"scin": 20, "ddi": 15, "hamisic_test": 20}

KNN_K = 10
FEWSHOT_K = 10
FEWSHOT_DRAWS = 5


def ensure_dirs() -> None:
    for d in (UNIFIED_DIR, EMBED_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
