"""Regenerate Figure 2 from ``results/purity.json``.

    python -m dermgap.figures --granularity fine
    python -m dermgap.figures --granularity category   # matched-granularity check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURE_DIR, MODEL_DISPLAY_NAMES, MODELS, RESULTS_DIR

IN_DOMAIN_KEY = "hamisic_test_mapped_label"
SCIN_KEYS = {"fine": "scin_label", "category": "scin_category"}


def plot_purity(purity: dict, granularity: str, out_path: Path) -> None:
    scin_key = SCIN_KEYS[granularity]
    in_domain = [purity[IN_DOMAIN_KEY][m]["lift"] for m in MODELS]
    out_domain = [purity[scin_key][m]["lift"] for m in MODELS]

    n_in = purity[IN_DOMAIN_KEY][MODELS[0]]["n_classes"]
    n_out = purity[scin_key][MODELS[0]]["n_classes"]

    x = np.arange(len(MODELS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=200)

    b1 = ax.bar(x - width / 2, in_domain, width,
                label=f"In-domain ({n_in} classes)", color="#4372A5")
    b2 = ax.bar(x + width / 2, out_domain, width,
                label=f"SCIN, out-of-domain ({n_out} classes)", color="#B4413F")

    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=8)

    ax.set_ylabel("kNN neighbor-purity lift over chance", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in MODELS], fontsize=9)
    ax.set_ylim(0, max(in_domain + out_domain) * 1.22)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} and {out_path.with_suffix('.pdf')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate Figure 2.")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "purity.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--granularity", choices=["fine", "category"], default="fine")
    args = ap.parse_args()

    purity = json.loads(args.results.read_text())
    out = args.out or FIGURE_DIR / f"fig2_purity_{args.granularity}.png"
    plot_purity(purity, args.granularity, out)


if __name__ == "__main__":
    main()
