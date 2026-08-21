"""Fine-tune the cancer baseline (ResNet-50) on pooled HAM10000 + ISIC 2019.

Reproduces the in-domain reference model: 8-class neoplastic taxonomy,
class-weighted cross-entropy, AdamW, cosine schedule, 15 epochs, grouped
70/15/15 split. Reported validation balanced accuracy: 0.673.

Checkpointing is per-epoch and resumable, because the original runs were done
in free-tier Kaggle sessions that can be interrupted.

    python -m dermgap.train_baseline --epochs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import CHECKPOINT_DIR, SEED, UNIFIED_DIR, ensure_dirs
from .models import IMAGENET_TF, TRAIN_TF, build_baseline_resnet
from .taxonomy import HAM_TO_8CLASS, ISIC_TO_8CLASS


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["image_path"].tolist()
        self.targets = df["y"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.targets[i]


def build_source_pool(unified_dir: Path) -> tuple[pd.DataFrame, dict]:
    ham = pd.read_csv(unified_dir / "ham10000_unified.csv")
    isic = pd.read_csv(unified_dir / "isic2019_unified.csv")
    ham["mapped_label"] = ham["label"].map(HAM_TO_8CLASS)
    isic["mapped_label"] = isic["label"].map(ISIC_TO_8CLASS)

    cols = ["image_path", "mapped_label", "patient_id", "source_dataset"]
    combined = pd.concat([ham[cols], isic[cols]], ignore_index=True)
    combined = combined[combined["mapped_label"].notna()].reset_index(drop=True)

    classes = sorted(combined["mapped_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    combined["y"] = combined["mapped_label"].map(class_to_idx)
    # Namespace the grouping key so the two sources cannot collide.
    combined["pid"] = combined["source_dataset"] + "_" + combined["patient_id"].astype(str)
    return combined, class_to_idx


def split_70_15_15(df: pd.DataFrame, seed: int = SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df["pid"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["pid"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    for a, b in ((train_df, val_df), (train_df, test_df), (val_df, test_df)):
        assert not (set(a["pid"]) & set(b["pid"])), "group leakage between splits"
    return train_df, val_df, test_df


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    ys, preds = [], []
    for imgs, y in loader:
        preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        ys.extend(np.asarray(y))
    return balanced_accuracy_score(ys, preds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the cancer baseline.")
    ap.add_argument("--unified", type=Path, default=UNIFIED_DIR)
    ap.add_argument("--out", type=Path, default=CHECKPOINT_DIR)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    ensure_dirs()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    combined, class_to_idx = build_source_pool(args.unified)
    classes = list(class_to_idx)
    print("Classes:", classes)
    print("Source pool:", combined.shape)

    train_df, val_df, test_df = split_70_15_15(combined, args.seed)
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)} "
          f"(grouped by lesion, no leakage)")

    loaders = {
        "train": DataLoader(SkinDataset(train_df, TRAIN_TF), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True),
        "val": DataLoader(SkinDataset(val_df, IMAGENET_TF), batch_size=args.batch_size,
                          shuffle=False, num_workers=args.workers, pin_memory=True),
        "test": DataLoader(SkinDataset(test_df, IMAGENET_TF), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.workers, pin_memory=True),
    }

    model = build_baseline_resnet(len(classes)).to(device)

    counts = train_df["y"].value_counts().sort_index().values
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = args.out / "checkpoint.pt"
    best_path = args.out / "baseline_resnet50_best.pt"
    start_epoch, best_val = 0, 0.0

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_val = ckpt["epoch"] + 1, ckpt["best_val_bacc"]
        print(f"Resumed at epoch {start_epoch} (best val {best_val:.3f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for imgs, y in loaders["train"]:
            imgs, y = imgs.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        val_bacc = evaluate(model, loaders["val"], device)
        print(f"Epoch {epoch + 1:2d}/{args.epochs} | "
              f"train_loss={running / len(loaders['train']):.3f} | val_bacc={val_bacc:.3f}")

        if val_bacc > best_val:
            best_val = val_bacc
            torch.save(model.state_dict(), best_path)
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_bacc": best_val}, ckpt_path)

    print(f"\nBest validation balanced accuracy: {best_val:.3f}")

    args.unified.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.unified / "baseline_train.csv", index=False)
    val_df.to_csv(args.unified / "baseline_val.csv", index=False)
    test_df.to_csv(args.unified / "baseline_test.csv", index=False)
    with open(args.unified / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Saved splits and class mapping to {args.unified}")


if __name__ == "__main__":
    main()
