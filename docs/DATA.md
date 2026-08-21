# Datasets

Four public datasets. **None are redistributed here.** Each carries its own license and access conditions; DDI and SCIN in particular require you to agree to terms before download. Obtain them yourself and point the code at them.

| Dataset | Images | Role in the study | Access |
|---|---|---|---|
| HAM10000 | 10,015 | Source domain (with ISIC) | Open, Harvard Dataverse / Kaggle |
| ISIC 2019 | 25,331 | Source domain | Open, ISIC Archive / Kaggle |
| DDI | 656 | Isolates the **tone** effect | Registration required, Stanford AIMI |
| SCIN | 6,517 | Isolates the **distribution** effect | Open, Google Research / GCS |

---

## Where to get them

**HAM10000** — Tschandl, Rosendahl & Kittler, *Scientific Data* 5:180161 (2018).
<https://doi.org/10.7910/DVN/DBW86T> (also on Kaggle as `kmader/skin-cancer-mnist-ham10000`)

**ISIC 2019** — Codella et al., arXiv:1902.03368; Tschandl et al.; Combalia et al.
<https://challenge.isic-archive.com/data/#2019> (also on Kaggle as `andrewmvd/isic-2019`)

**DDI (Diverse Dermatology Images)** — Daneshjou et al., *Science Advances* 8(31):eabq6147 (2022).
<https://ddi-dataset.github.io/> — requires accepting the data use agreement.

**SCIN (Skin Condition Image Network)** — Ward et al., arXiv:2402.18545 (2024).
<https://github.com/google-research-datasets/scin>

---

## Expected layout

The loaders in `dermgap/datasets.py` expect these structures. Override each root with an environment variable.

### HAM10000 — `DERMGAP_HAM10000`

```
ham10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   └── ISIC_0024306.jpg ...
└── HAM10000_images_part_2/
    └── ISIC_0029306.jpg ...
```

Columns used: `image_id`, `dx`, `lesion_id`.

### ISIC 2019 — `DERMGAP_ISIC2019`

```
isic2019/
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Training_Metadata.csv
└── ISIC_2019_Training_Input/
    └── ISIC_2019_Training_Input/         # some mirrors double-nest this
        └── ISIC_0000000.jpg ...
```

The loader tries the double-nested path first and falls back to the flat one. Columns used: `image`, the one-hot label columns (`MEL NV BCC AK BKL DF VASC SCC UNK`), and `lesion_id` from the metadata file.

Note: `UNK` is all-zero in the 2019 *training* ground truth, so no rows are lost to it — the pooled source set is the full 35,346 images.

### DDI — `DERMGAP_DDI`

```
ddi/
├── ddi_metadata.csv
└── images/
    └── 000001.png ...
```

Columns used: `DDI_file`, `DDI_ID`, `disease`, `malignant`, `skin_tone`.

`skin_tone` is coded `12` (Fitzpatrick I–II), `34` (III–IV), `56` (V–VI).

**`DDI_ID` is unique per image** — 656 IDs for 656 images. DDI distributes no patient identifier, so DDI splits are image-level, not patient-level. This is stated in the paper and in `docs/REVIEW_NOTES.md`.

### SCIN — `DERMGAP_SCIN`

```
scin/
├── scin_cases.csv
├── scin_labels.csv
└── images/
    └── -3205742176803893704.png ...
```

Columns used: `case_id`, `image_1_path`, `image_2_path`, `image_3_path`, `fitzpatrick_skin_type`, `weighted_skin_condition_label`.

Two quirks the loader handles:

1. The `image_*_path` columns store paths of the form `dataset/images/<file>`. Only the basename is used; files are read from `<root>/images/<file>`.
2. `weighted_skin_condition_label` is a stringified dict of `{condition: weight}`. The loader takes the arg-max as the label.

**Cases with an empty label dict are dropped: 5,033 → 3,061 cases (39.2%).** Combined with the three image columns and one missing file, this yields 6,517 images. This is a large, non-random filter — see §6 of `docs/REVIEW_NOTES.md`.

---

## Configuring paths

```bash
export DERMGAP_HAM10000=/data/ham10000
export DERMGAP_ISIC2019=/data/isic2019
export DERMGAP_DDI=/data/ddi
export DERMGAP_SCIN=/data/scin

export DERMGAP_ARTIFACTS=./artifacts      # unified tables, checkpoints, embeddings
export DERMGAP_RESULTS=./results
export DERMGAP_FIGURES=./figures
```

Defaults point at the Kaggle mount layout used for the original runs, so on Kaggle nothing needs setting. Then:

```bash
python -m dermgap.datasets
```

This writes `{ham10000,isic2019,ddi,scin}_unified.csv` into `$DERMGAP_ARTIFACTS/unified/`. Any dataset whose root is missing is skipped with a message rather than crashing, so you can build incrementally.

---

## Disk and compute

| Stage | Storage | Compute |
|---|---|---|
| Raw images | ~15 GB (ISIC 2019 dominates) | — |
| Unified tables | <10 MB | ~2 min CPU |
| Baseline fine-tuning | ~100 MB checkpoints | 1 Kaggle T4 session, 15 epochs |
| Frozen embeddings | ~350 MB (4 models × 3 sets) | 1 Kaggle T4 session, resumable |
| Analyses + figures | <5 MB | ~5 min CPU |

Feature extraction skips any `(model, dataset)` pair whose `.npy` already exists, so an interrupted free-tier session can be restarted without losing work.

---

## Model weights

Downloaded automatically from Hugging Face on first use:

- DermLIP — `redlessone/DermLIP_ViT-B-16` (via `open_clip`)
- MONET — `chanwkim/monet`
- DINOv3 — `facebook/dinov3-vitb16-pretrain-lvd1689m` — **gated**; request access, then `huggingface-cli login` or export `HF_TOKEN`

The cancer baseline is trained locally by `dermgap.train_baseline`.
