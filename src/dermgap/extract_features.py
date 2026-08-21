"""Extract frozen embeddings for every (model, dataset) pair.

Writes ``{model}_{dataset}.npy`` plus one ``meta_{dataset}.csv`` per dataset
into the embeddings directory. Resumable: an existing .npy is never recomputed,
so an interrupted free-tier session can simply be restarted.

    python -m dermgap.extract_features --models all --datasets all

DINOv3 is a gated repository. Authenticate first with ``huggingface-cli login``
or by exporting ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import CHECKPOINT_DIR, EMBED_DIR, EVAL_DATASETS, MODELS, UNIFIED_DIR, ensure_dirs
from .models import MODEL_LOADERS

# Which unified table backs each evaluation split.
SOURCE_TABLE = {
    "ddi": "ddi_unified.csv",
    "scin": "scin_unified.csv",
    "hamisic_test": "baseline_test.csv",
}


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess):
        self.paths = df["image_path"].tolist()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.preprocess(img), i


def load_eval_table(unified_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(unified_dir / SOURCE_TABLE[name])
    n0 = len(df)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    if n0 != len(df):
        print(f"  {name}: dropped {n0 - len(df)} missing-image rows, {len(df)} remain")
    return df


@torch.inference_mode()
def extract(forward_fn, preprocess, df, device, batch_size=64, workers=2) -> np.ndarray:
    loader = DataLoader(ImageDataset(df, preprocess), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True)
    out = [None] * len(df)
    for batch, idxs in loader:
        feats = forward_fn(batch.to(device)).float().cpu().numpy()
        for j, idx in enumerate(idxs.numpy()):
            out[idx] = feats[j]
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frozen features.")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--unified", type=Path, default=UNIFIED_DIR)
    ap.add_argument("--out", type=Path, default=EMBED_DIR)
    ap.add_argument("--baseline-checkpoint", type=Path,
                    default=CHECKPOINT_DIR / "baseline_resnet50_best.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--official-preprocess", action="store_true",
                    help="Use canonical resize+center-crop for MONET/DINOv3 "
                         "instead of the paper's squash-resize (robustness check).")
    args = ap.parse_args()

    ensure_dirs()
    args.out.mkdir(parents=True, exist_ok=True)

    model_names = MODELS if args.models == ["all"] else args.models
    dataset_names = EVAL_DATASETS if args.datasets == ["all"] else args.datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading evaluation tables...")
    tables = {name: load_eval_table(args.unified, name) for name in dataset_names}

    for model_name in model_names:
        targets = [d for d in dataset_names
                   if not (args.out / f"{model_name}_{d}.npy").exists()]
        if not targets:
            print(f"[{model_name}] already complete, skipping.")
            continue

        print(f"\n=== Loading {model_name} ===")
        forward_fn, preprocess = MODEL_LOADERS[model_name](
            device, checkpoint=str(args.baseline_checkpoint),
            official=args.official_preprocess)

        for ds in targets:
            df = tables[ds]
            print(f"  {ds}: extracting {len(df)} images...")
            emb = extract(forward_fn, preprocess, df, device,
                          args.batch_size, args.workers)
            np.save(args.out / f"{model_name}_{ds}.npy", emb)

            meta_path = args.out / f"meta_{ds}.csv"
            if not meta_path.exists():
                df.to_csv(meta_path, index=False)
            print(f"    saved {emb.shape} -> {model_name}_{ds}.npy")

        del forward_fn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nFeature extraction complete. Contents of", args.out)
    for f in sorted(os.listdir(args.out)):
        print("  ", f)


if __name__ == "__main__":
    main()
