# Disease Burden over Skin Tone: Decomposing the Dermatology-AI Generalization Gap

Code and evaluation protocol for the paper *Disease Burden over Skin Tone: Decomposing the Dermatology-AI Generalization Gap*.

Dermatology AI models are trained mostly on light-skinned, cancer-focused image collections but are increasingly proposed for resource-constrained settings whose patients differ along **two confounded axes at once**: skin tone and disease distribution. When these models underperform, the failure is usually attributed to skin-tone underrepresentation. This repository disentangles the two factors using only public datasets and free-tier compute.

**Headline result.** Disease-distribution shift dominates. A cancer-trained baseline drops from 0.62 balanced accuracy in-domain to 0.21 on unfamiliar conditions, while the within-disease skin-tone gap is small (0.10–0.18) and inconsistent in direction. A label-free analysis shows the collapse is representational rather than an artifact of disjoint label spaces, and cheap adaptation recovers performance only in proportion to the structure already present in the frozen features.

---

## Repository layout

```
.
├── src/dermgap/            # importable package; every script is a CLI
│   ├── config.py           # paths, model list, seeds, thresholds
│   ├── taxonomy.py         # canonical label mappings (single source of truth)
│   ├── datasets.py         # build unified metadata tables
│   ├── models.py           # the four frozen feature extractors
│   ├── train_baseline.py   # fine-tune the cancer baseline
│   ├── extract_features.py # frozen embeddings for every (model, dataset)
│   ├── analysis.py         # probes, bootstrap CIs, kNN purity
│   ├── evaluate.py         # runs all three analyses, writes results/
│   ├── null_calibration.py # how large is a tone gap of 0.14, really?
│   └── figures.py          # regenerate Figure 2
├── notebooks/              # the original Kaggle notebooks, outputs intact
├── results/                # recorded numbers + regenerated JSON
├── tests/                  # data-free tests for the analysis primitives
└── docs/
    ├── DATA.md             # how to obtain and lay out the four datasets
    └── REVIEW_NOTES.md     # known issues and camera-ready checklist
```

The notebooks under `notebooks/` are the **canonical record**: they are the code that produced the numbers in the paper, with their outputs preserved. The `src/dermgap` package is a cleaned, deduplicated, tested port of the same logic, intended for anyone rerunning or extending the work. Where the two could drift, the notebooks win; `results/original_results.json` records exactly what the notebooks produced.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/Nirajan995/dermatology-generalization-gap.git
cd dermatology-generalization-gap

python -m venv .venv && source .venv/bin/activate    # Python 3.10+
pip install -e ".[full]"
```

For the CPU-only analysis (probes, purity, adaptation, figures) you do not need PyTorch:

```bash
pip install -e .
```

Verify the install:

```bash
pytest -q          # 22 tests, no data required, runs in ~2s
```

### 2. Get the data

Four public datasets. See **[docs/DATA.md](docs/DATA.md)** for download links, access requirements, and the expected directory layout.

| Dataset | Images | Role |
|---|---|---|
| HAM10000 | 10,015 | Source domain (with ISIC) |
| ISIC 2019 | 25,331 | Source domain |
| DDI | 656 | Isolates the **tone** effect |
| SCIN | 6,517 | Isolates the **distribution** effect |

Point the code at them with environment variables:

```bash
export DERMGAP_HAM10000=/path/to/ham10000
export DERMGAP_ISIC2019=/path/to/isic2019
export DERMGAP_DDI=/path/to/ddi
export DERMGAP_SCIN=/path/to/scin
export DERMGAP_ARTIFACTS=./artifacts        # where tables, weights, embeddings go
```

### 3. Run the pipeline

```bash
bash scripts/run_all.sh
```

Or step by step:

```bash
# (a) Build unified metadata tables                                    ~2 min, CPU
python -m dermgap.datasets

# (b) Fine-tune the cancer baseline                       ~1 GPU session (T4, 15 epochs)
python -m dermgap.train_baseline --epochs 15

# (c) Extract frozen features, all 4 models x 3 eval sets      ~1 GPU session, resumable
huggingface-cli login          # DINOv3 is a gated repo
python -m dermgap.extract_features --models all --datasets all

# (d) All three analyses                                             ~5 min, CPU only
python -m dermgap.evaluate --analysis all

# (e) Figure 2 and the tone-gap null calibration                            seconds, CPU
python -m dermgap.figures --granularity fine
python -m dermgap.null_calibration
```

Steps (b) and (c) each fit in a single free-tier Kaggle T4 session and both checkpoint per unit of work, so an interrupted session can simply be restarted. Everything downstream is CPU-only and finishes in minutes. At inference each image needs one forward pass through a frozen encoder plus a linear probe, so no specialized hardware is required.

### 4. Check the reproduction

`python -m dermgap.evaluate` writes `decomposition.json`, `purity.json`, `adaptation.json` and a human-readable `summary.md` into `results/`. Compare against `results/original_results.json`, which records what the original runs produced.

---

## The three analyses

**1. Decomposition (Table 2).** L2-regularised logistic probes on standardized frozen features, grouped 70/30 split, balanced accuracy. The tone effect comes from DDI (diagnosis broadly fixed, Fitzpatrick tone varying, binary malignant/benign target); the distribution effect comes from SCIN (tone-diverse, dominated by non-neoplastic conditions).

**2. Label-free representation quality (Figure 2).** k=10 cosine neighbour purity, reported as *lift* over the random-neighbour floor `sum_i p_i^2` so class imbalance cannot inflate it. This asks whether frozen features geometrically separate conditions **before any classifier is trained**, which is what distinguishes a genuine representational deficit from a merely missing output head.

**3. Low-compute adaptation (Table 3).** Full probe vs few-shot probe (10 examples per class, averaged over 5 draws) vs raw-versus-standardized features, all without updating any model weights.

All confidence intervals are 1000-sample bootstraps. Seed is 42 throughout.

---

## Reading the numbers responsibly

Three points that matter for anyone building on this work. All three are handled in the code and documented in [docs/REVIEW_NOTES.md](docs/REVIEW_NOTES.md).

**Absolute accuracies are low by design.** Features are frozen and nothing is fine-tuned, so the probes measure linear separability of fixed representations, not the ceiling of a trained model — the same baseline reaches 0.673 when fine-tuned end to end. The argument rests on *relative* drops across domains and on the label-free analysis, which needs no classifier at all.

**The tone gap is not distinguishable from zero.** `max - min` across three Fitzpatrick groups is biased upward: the maximum of three noisy estimates exceeds the minimum even when all groups share the same true accuracy, and the bootstrap interval therefore never contains zero by construction. `python -m dermgap.null_calibration` simulates the null at DDI's actual group sizes (55 / 65 / 77): with **no tone effect at all**, the expected gap is ≈0.09 with a 95% range of ≈0.02–0.20. Every gap reported in the paper falls inside that range. The honest statement is that the tone effect is not resolvable at this sample size — which is conservative, and supports rather than weakens the paper's conclusion.

**"Patient-level split" means different things per dataset.** The grouping key is the finest identifier each dataset ships, and it is not always a patient:

| Dataset | Grouping key | Actual granularity |
|---|---|---|
| HAM10000 | `lesion_id` | lesion (no patient ID is distributed) |
| ISIC 2019 | `lesion_id`, falling back to image id | lesion |
| DDI | `DDI_ID` | **image** — 656 IDs for 656 images, so this is an image-level split |
| SCIN | `case_id` | case (one crowdsourced submission) |

Grouping still prevents the leakage it can prevent (multiple images of one lesion or one SCIN case never straddle the split), but DDI cannot be split at the patient level with the released metadata.

---

## Models

| Key | Model | Dim | Role |
|---|---|---|---|
| `resnet_baseline` | ResNet-50 fine-tuned on HAM10000+ISIC | 2048 | Cancer-specialized baseline |
| `dermlip` | [DermLIP ViT-B/16](https://huggingface.co/redlessone/DermLIP_ViT-B-16) | 512 | Dermatology vision-language FM |
| `monet` | [MONET](https://huggingface.co/chanwkim/monet) | 1024 | Dermatology vision-language FM |
| `dinov3` | [DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 768 | General-purpose control (no dermatology knowledge) |

No model weights are updated during evaluation. DINOv3 is gated on Hugging Face — request access and authenticate before running extraction.

---

## Citation

```bibtex
@inproceedings{kunwor2026diseaseburden,
  title     = {Disease Burden over Skin Tone: Decomposing the Dermatology-AI Generalization Gap},
  author    = {Kunwor, Nirajan and Poudel, Sanjaya and Trinh, Quoc-Huy and
               Arfat, Jahidul and Gaire, Sunil Kumar},
  year      = {2026}
}
```

Please also cite the underlying datasets (HAM10000, ISIC 2019, DDI, SCIN) and models (DermLIP, MONET, DINOv3). Full references are in the paper.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The datasets carry their own licenses and access conditions, which this repository does not alter or redistribute — see [docs/DATA.md](docs/DATA.md).

## Intended use

This is a research auditing tool, not a clinical device. Nothing here is validated for diagnostic use. DDI and SCIN are US-sourced datasets used as controlled proxies to isolate each factor; they are not collected in resource-constrained settings, so conclusions about deployment there are inferential and await validation on genuinely RCS-collected cohorts.
