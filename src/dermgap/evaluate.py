"""Run all three analyses and write every number reported in the paper.

    python -m dermgap.evaluate --analysis all

Outputs (into ``results/``):

    decomposition.json  Table 2 — in-domain vs SCIN vs DDI tone gap
    purity.json         Figure 2 — label-free kNN neighbour-purity lift
    adaptation.json     Table 3 — full probe vs few-shot vs raw features
    summary.md          the three tables in markdown, for pasting into the paper

Everything here is CPU-only and runs in a few minutes once the embeddings
exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .analysis import (
    bootstrap_ci,
    bootstrap_group_gap,
    filter_rare_classes,
    fit_fewshot_probe,
    fit_linear_probe,
    knn_purity,
)
from .config import (
    EMBED_DIR,
    FEWSHOT_DRAWS,
    FEWSHOT_K,
    KNN_K,
    LABEL_COL,
    MIN_CLASS_COUNT_PROBE,
    MIN_CLASS_COUNT_PURITY,
    MODELS,
    RESULTS_DIR,
    ensure_dirs,
)
from .taxonomy import TONE_MAP, map_scin_category


def load_pair(embed_dir: Path, model: str, dataset: str):
    emb = np.load(embed_dir / f"{model}_{dataset}.npy")
    meta = pd.read_csv(embed_dir / f"meta_{dataset}.csv")
    if len(emb) != len(meta):
        raise ValueError(f"length mismatch for {model}/{dataset}: "
                         f"{len(emb)} embeddings vs {len(meta)} metadata rows")
    if dataset == "scin":
        meta["category"] = meta["label"].apply(map_scin_category)
    return emb, meta


# ---------------------------------------------------------------------------
# Analysis 1 — decomposition (Table 2)
# ---------------------------------------------------------------------------

def probe_dataset(emb, meta, label_col, group_col, min_count):
    mask = filter_rare_classes(meta[label_col].values, min_count)
    emb_f = emb[mask]
    meta_f = meta[mask].reset_index(drop=True)
    _, y_te, pred = fit_linear_probe(
        emb_f, meta_f[label_col].values, meta_f[group_col].values)
    point, boot_mean, lo, hi = bootstrap_ci(y_te, pred, balanced_accuracy_score)
    return {
        "balanced_accuracy": round(point, 4),
        "bootstrap_mean": round(boot_mean, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_classes": int(meta_f[label_col].nunique()),
        "n_images": len(meta_f),
    }


def run_decomposition(embed_dir: Path) -> dict:
    out = {"in_domain": {}, "scin_fine": {}, "scin_category": {}, "ddi_tone": {}}

    print("\n" + "=" * 72)
    print("ANALYSIS 1 — DECOMPOSITION (Table 2)")
    print("=" * 72)
    header = f"{'Model':<17}{'In-domain':<12}{'SCIN cat':<12}{'SCIN fine':<12}{'Tone gap':<12}"
    print(header)
    print("-" * 72)

    for model in MODELS:
        emb_id, meta_id = load_pair(embed_dir, model, "hamisic_test")
        out["in_domain"][model] = probe_dataset(
            emb_id, meta_id, LABEL_COL["hamisic_test"], "pid", MIN_CLASS_COUNT_PROBE)

        emb_sc, meta_sc = load_pair(embed_dir, model, "scin")
        out["scin_fine"][model] = probe_dataset(
            emb_sc, meta_sc, "label", "patient_id", MIN_CLASS_COUNT_PROBE)
        out["scin_category"][model] = probe_dataset(
            emb_sc, meta_sc, "category", "patient_id", 0)

        # DDI: binary malignant target, accuracy gap across Fitzpatrick groups.
        emb_dd, meta_dd = load_pair(embed_dir, model, "ddi")
        y = meta_dd["malignant"].astype(int).values
        te, y_te, pred = fit_linear_probe(
            emb_dd, y, meta_dd["patient_id"].values)
        tone_te = pd.Series(meta_dd["skin_tone"].values[te]).map(TONE_MAP).values
        observed, boot_mean, lo, hi, per_group = bootstrap_group_gap(y_te, pred, tone_te)
        out["ddi_tone"][model] = {
            "overall_balanced_accuracy": round(balanced_accuracy_score(y_te, pred), 4),
            "per_group_accuracy": {k: round(v, 4) for k, v in per_group.items()},
            "per_group_n": {k: int((tone_te == k).sum()) for k in per_group},
            "gap_observed": round(observed, 4),
            "gap_bootstrap_mean": round(boot_mean, 4),
            "gap_ci_low": round(lo, 4),
            "gap_ci_high": round(hi, 4),
        }

        print(f"{model:<17}"
              f"{out['in_domain'][model]['balanced_accuracy']:<12.3f}"
              f"{out['scin_category'][model]['balanced_accuracy']:<12.3f}"
              f"{out['scin_fine'][model]['balanced_accuracy']:<12.3f}"
              f"{observed:<12.3f}")

    out["_notes"] = {
        "tone_gap_metric": "max-minus-min of per-Fitzpatrick-group plain accuracy "
                           "on the binary malignant task (NOT balanced accuracy)",
        "tone_gap_caveat": "max-min of noisy per-group estimates is biased upward; "
                           "the interval is descriptive and is not a test against zero",
        "grouping": "hamisic_test grouped by lesion, scin by case, ddi by image "
                    "(DDI ships no patient identifier)",
    }
    return out


# ---------------------------------------------------------------------------
# Analysis 2 — label-free representation quality (Figure 2)
# ---------------------------------------------------------------------------

def run_purity(embed_dir: Path) -> dict:
    print("\n" + "=" * 72)
    print("ANALYSIS 2 — LABEL-FREE kNN NEIGHBOUR PURITY (Figure 2)")
    print("=" * 72)

    specs = [
        ("hamisic_test", "mapped_label", MIN_CLASS_COUNT_PURITY["hamisic_test"]),
        ("scin", "label", MIN_CLASS_COUNT_PURITY["scin"]),
        ("scin", "category", 50),
        ("ddi", "label", MIN_CLASS_COUNT_PURITY["ddi"]),
    ]

    out = {}
    for dataset, label_col, min_count in specs:
        key = f"{dataset}_{label_col}"
        out[key] = {}
        meta = load_pair(embed_dir, MODELS[0], dataset)[1]
        mask = filter_rare_classes(meta[label_col].values, min_count)
        labels = meta[label_col].values[mask]

        print(f"\n{dataset.upper()} — label='{label_col}', min_count={min_count}")
        for model in MODELS:
            emb = np.load(embed_dir / f"{model}_{dataset}.npy")[mask]
            res = knn_purity(emb, labels, k=KNN_K)
            out[key][model] = {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in res.items()}
            print(f"  {model:<17}purity={res['purity']:.3f} "
                  f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  "
                  f"chance={res['chance']:.3f}  lift={res['lift']:+.3f}")

    out["_notes"] = {
        "metric": "fraction of each image's k=10 cosine nearest neighbours sharing "
                  "its label, minus the random-neighbour floor sum_i p_i^2",
        "granularity_warning": "Figure 2 pairs in-domain purity at 8 classes with "
                               "SCIN purity at 47 classes. Lift is chance-corrected, "
                               "but class counts differ; scin_category (5-6 classes) "
                               "is the matched-granularity robustness check.",
        "uncontrolled": "purity is not controlled for embedding dimensionality "
                        "(2048 / 512 / 1024 / 768)",
    }
    return out


# ---------------------------------------------------------------------------
# Analysis 3 — low-compute adaptation (Table 3)
# ---------------------------------------------------------------------------

def run_adaptation(embed_dir: Path, purity_results: dict | None = None) -> dict:
    print("\n" + "=" * 72)
    print("ANALYSIS 3 — LOW-COMPUTE ADAPTATION ON SCIN CATEGORIES (Table 3)")
    print("=" * 72)
    print(f"{'Model':<17}{'raw':<10}{'standardized':<15}{'few-shot k=10':<16}")
    print("-" * 72)

    out = {}
    for model in MODELS:
        emb, meta = load_pair(embed_dir, model, "scin")
        y = meta["category"].values
        groups = meta["patient_id"].values

        _, y_te, pred = fit_linear_probe(emb, y, groups, standardize=True)
        std_bacc = balanced_accuracy_score(y_te, pred)

        _, y_te_raw, pred_raw = fit_linear_probe(emb, y, groups, standardize=False)
        raw_bacc = balanced_accuracy_score(y_te_raw, pred_raw)

        fs_mean, fs_std = fit_fewshot_probe(
            emb, y, groups, k=FEWSHOT_K, n_draws=FEWSHOT_DRAWS, standardize=True)

        out[model] = {
            "raw": round(raw_bacc, 4),
            "standardized": round(std_bacc, 4),
            "fewshot_k10_mean": round(fs_mean, 4),
            "fewshot_k10_std": round(fs_std, 4),
            "standardization_gain": round(std_bacc - raw_bacc, 4),
        }
        print(f"{model:<17}{raw_bacc:<10.3f}{std_bacc:<15.3f}{fs_mean:<16.3f}")

    # Per-category recall for the cancer baseline: where exactly does it fail?
    emb, meta = load_pair(embed_dir, "resnet_baseline", "scin")
    _, y_te, pred = fit_linear_probe(
        emb, meta["category"].values, meta["patient_id"].values)
    recalls = {}
    print("\nPer-category recall — cancer baseline:")
    for cat in sorted(set(y_te)):
        mask = y_te == cat
        recalls[cat] = {"n": int(mask.sum()), "recall": round(float((pred[mask] == cat).mean()), 4)}
        print(f"  {cat:<20}n={mask.sum():<5}recall={recalls[cat]['recall']:.3f}")
    out["_cancer_baseline_per_category_recall"] = recalls

    # Link back to Analysis 2.
    if purity_results:
        lift = {m: purity_results["scin_label"][m]["lift"] for m in MODELS}
        acc = [out[m]["standardized"] for m in MODELS]
        r = float(np.corrcoef([lift[m] for m in MODELS], acc)[0, 1])
        acc_raw = [out[m]["raw"] for m in MODELS]
        r_raw = float(np.corrcoef([lift[m] for m in MODELS], acc_raw)[0, 1])
        out["_purity_vs_recovery"] = {
            "purity_lift_scin_fine": lift,
            "pearson_r_vs_standardized_probe": round(r, 3),
            "pearson_r_vs_raw_probe": round(r_raw, 3),
            "caveat": "n=4 models; descriptive, not a significance test",
        }
        print(f"\nCorrelation, SCIN purity lift vs full probe: r = {r:.3f} "
              f"(raw features: r = {r_raw:.3f}); n=4, descriptive only")

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_summary(results: dict, path: Path) -> None:
    lines = ["# Reproduced results", "",
             "Regenerated by `python -m dermgap.evaluate`. "
             "Numbers should match the paper to the reported precision.", ""]

    dec = results.get("decomposition")
    if dec:
        lines += ["## Table 2 — decomposition", "",
                  "| Model | In-domain | SCIN (category) | SCIN (fine) | DDI tone gap [95% CI] |",
                  "|---|---|---|---|---|"]
        for m in MODELS:
            t = dec["ddi_tone"][m]
            lines.append(
                f"| {m} | {dec['in_domain'][m]['balanced_accuracy']:.2f} "
                f"| {dec['scin_category'][m]['balanced_accuracy']:.2f} "
                f"| {dec['scin_fine'][m]['balanced_accuracy']:.2f} "
                f"| {t['gap_observed']:.2f} ({t['gap_ci_low']:.2f}–{t['gap_ci_high']:.2f}) |")
        lines.append("")
        lines.append("Tone gap is max-minus-min of per-group *plain* accuracy on the "
                     "binary malignant task; see caveat in `decomposition.json`.")
        lines.append("")

    pur = results.get("purity")
    if pur:
        lines += ["## Figure 2 — kNN neighbour-purity lift over chance", "",
                  "| Model | In-domain (8 cls) | SCIN fine (47 cls) | SCIN category (5-6 cls) |",
                  "|---|---|---|---|"]
        for m in MODELS:
            lines.append(
                f"| {m} | {pur['hamisic_test_mapped_label'][m]['lift']:+.2f} "
                f"| {pur['scin_label'][m]['lift']:+.2f} "
                f"| {pur['scin_category'][m]['lift']:+.2f} |")
        lines.append("")

    ada = results.get("adaptation")
    if ada:
        lines += ["## Table 3 — low-compute adaptation on SCIN categories", "",
                  "| Model | Full probe (std) | Few-shot k=10 | Raw features | SCIN purity lift |",
                  "|---|---|---|---|---|"]
        for m in MODELS:
            lift = (results.get("purity", {}).get("scin_label", {}).get(m, {}).get("lift"))
            lift_s = f"{lift:+.2f}" if lift is not None else "n/a"
            lines.append(
                f"| {m} | {ada[m]['standardized']:.2f} | {ada[m]['fewshot_k10_mean']:.2f} "
                f"| {ada[m]['raw']:.2f} | {lift_s} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the paper's analyses.")
    ap.add_argument("--embeddings", type=Path, default=EMBED_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--analysis", choices=["all", "decomposition", "purity", "adaptation"],
                    default="all")
    args = ap.parse_args()

    ensure_dirs()
    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.analysis in ("all", "decomposition"):
        results["decomposition"] = run_decomposition(args.embeddings)
        (args.out / "decomposition.json").write_text(
            json.dumps(results["decomposition"], indent=2))

    if args.analysis in ("all", "purity", "adaptation"):
        results["purity"] = run_purity(args.embeddings)
        (args.out / "purity.json").write_text(json.dumps(results["purity"], indent=2))

    if args.analysis in ("all", "adaptation"):
        results["adaptation"] = run_adaptation(args.embeddings, results.get("purity"))
        (args.out / "adaptation.json").write_text(json.dumps(results["adaptation"], indent=2))

    if args.analysis == "all":
        write_summary(results, args.out / "summary.md")
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
