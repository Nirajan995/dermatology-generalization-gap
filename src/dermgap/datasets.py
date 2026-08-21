"""Build unified metadata tables for all four datasets.

Every dataset is reduced to a common schema::

    image_path, label, patient_id, source_dataset [, extra columns]

``patient_id`` is the grouping key used by every split in this repository.
IMPORTANT: the granularity of that key differs by dataset and is *not* always
a patient (see docs/DATA.md and the note in the README):

    HAM10000  -> lesion_id      (lesion-level; HAM10000 ships no patient ID)
    ISIC 2019 -> lesion_id, falling back to image id when absent (lesion-level)
    DDI       -> DDI_ID          (image-level; DDI ships no patient ID)
    SCIN      -> case_id         (case-level; one crowdsourced submission)

Run as a module::

    python -m dermgap.datasets --out artifacts/unified
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import pandas as pd

from .config import DATASET_PATHS, UNIFIED_DIR

# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def build_ham10000(root: Path) -> pd.DataFrame:
    meta = pd.read_csv(root / "HAM10000_metadata.csv")

    def image_path(image_id: str):
        for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
            p = root / folder / f"{image_id}.jpg"
            if p.exists():
                return str(p)
        return None

    return pd.DataFrame({
        "image_path": meta["image_id"].apply(image_path),
        "label": meta["dx"],
        "patient_id": meta["lesion_id"],   # lesion-level grouping
        "source_dataset": "ham10000",
    })


def build_isic2019(root: Path) -> pd.DataFrame:
    labels = pd.read_csv(root / "ISIC_2019_Training_GroundTruth.csv")
    meta = pd.read_csv(root / "ISIC_2019_Training_Metadata.csv")

    label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    labels["label"] = labels[label_cols].idxmax(axis=1)
    merged = labels.merge(meta, on="image", how="left")

    img_dir = root / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
    if not img_dir.exists():  # some mirrors do not double-nest the folder
        img_dir = root / "ISIC_2019_Training_Input"

    return pd.DataFrame({
        "image_path": merged["image"].apply(lambda x: str(img_dir / f"{x}.jpg")),
        "label": merged["label"],
        "patient_id": merged["lesion_id"].fillna(merged["image"]),
        "source_dataset": "isic2019",
    })


def build_ddi(root: Path) -> pd.DataFrame:
    meta = pd.read_csv(root / "ddi_metadata.csv")
    return pd.DataFrame({
        "image_path": meta["DDI_file"].apply(lambda x: str(root / "images" / x)),
        "label": meta["disease"],
        "malignant": meta["malignant"],
        "skin_tone": meta["skin_tone"],      # 12 / 34 / 56
        "patient_id": meta["DDI_ID"],        # image-level; see module docstring
        "source_dataset": "ddi",
    })


def _parse_top_label(label_str):
    """SCIN ships a dict of {condition: weight}; take the arg-max."""
    try:
        d = ast.literal_eval(label_str)
        return max(d, key=d.get) if d else None
    except (ValueError, SyntaxError, TypeError):
        return None


def build_scin(root: Path) -> pd.DataFrame:
    cases = pd.read_csv(root / "scin_cases.csv")
    labels = pd.read_csv(root / "scin_labels.csv")
    merged = cases.merge(labels, on="case_id", how="inner")

    merged["parsed_label"] = merged["weighted_skin_condition_label"].apply(_parse_top_label)

    n_before = len(merged)
    merged = merged[merged["parsed_label"].notna()].copy()
    print(f"  SCIN: {n_before} cases -> {len(merged)} with a usable label "
          f"(dropped {n_before - len(merged)}, {(n_before - len(merged)) / n_before * 100:.1f}%)")

    rows = []
    for _, row in merged.iterrows():
        for col in ("image_1_path", "image_2_path", "image_3_path"):
            raw = row.get(col)
            if pd.notna(raw):
                rows.append({
                    # CSV stores "dataset/images/<file>"; files sit at <root>/images/<file>
                    "image_path": str(root / "images" / os.path.basename(raw)),
                    "label": row["parsed_label"],
                    "fitzpatrick_skin_type": row.get("fitzpatrick_skin_type"),
                    "patient_id": row["case_id"],   # case-level grouping
                    "source_dataset": "scin",
                })
    return pd.DataFrame(rows)


BUILDERS = {
    "ham10000": build_ham10000,
    "isic2019": build_isic2019,
    "ddi": build_ddi,
    "scin": build_scin,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drop_missing_images(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drop rows whose image file is absent. Reported, never silent."""
    n0 = len(df)
    df = df[df["image_path"].notna()]
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    dropped = n0 - len(df)
    if dropped:
        print(f"  {name}: dropped {dropped} missing-image rows "
              f"({dropped / n0 * 100:.2f}%), {len(df)} remain")
    else:
        print(f"  {name}: all {len(df)} images present")
    return df


def build_all(out_dir: Path = UNIFIED_DIR) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, builder in BUILDERS.items():
        root = DATASET_PATHS[name]
        if not root.exists():
            print(f"  {name}: SKIPPED (not found at {root})")
            continue
        print(f"Building {name} from {root}")
        df = drop_missing_images(builder(root), name)
        df.to_csv(out_dir / f"{name}_unified.csv", index=False)
        tables[name] = df

    print("\n" + "=" * 62)
    print("UNIFIED TABLES")
    print("=" * 62)
    for name, df in tables.items():
        print(f"{name:12s} rows={len(df):6d}  groups={df['patient_id'].nunique():6d}  "
              f"labels={df['label'].nunique():4d}")
    print(f"\nSaved to {out_dir}")
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified dataset tables.")
    ap.add_argument("--out", type=Path, default=UNIFIED_DIR)
    args = ap.parse_args()
    build_all(args.out)


if __name__ == "__main__":
    main()
