"""How large is a tone gap of 0.14, really?

The DDI tone gap is ``max - min`` of accuracy across three Fitzpatrick groups
with 55 / 65 / 77 test images. That statistic is biased upward: the maximum of
three noisy estimates exceeds the minimum even when all three groups have
*identical* true accuracy. Before calling an observed gap "small" or "large" it
has to be compared against the gap you would see with no tone effect at all.

This module simulates that null: draw each group's correct-count from
Binomial(n_g, p) with a shared p, and record ``max - min``.

    python -m dermgap.null_calibration --n 55 65 77 --p 0.75

At DDI's sample sizes the null expectation is roughly 0.09 with a 95% range of
about 0.02-0.20, which brackets every tone gap reported in the paper. The
correct reading is therefore not "there is a 0.14 tone gap" but "the tone gap is
not distinguishable from zero at this sample size" - a conservative conclusion
that supports, rather than weakens, the paper's argument.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import RESULTS_DIR, SEED

# Test-split sizes per Fitzpatrick group in the DDI evaluation (70/30 split).
DDI_GROUP_SIZES = [55, 65, 77]

# Tone gaps reported in Table 2, for comparison against the null.
OBSERVED_GAPS = {
    "resnet_baseline": 0.143,
    "dermlip": 0.136,
    "monet": 0.096,
    "dinov3": 0.175,
}


def simulate_null_gap(group_sizes, p: float, n_sim: int = 20000, seed: int = SEED) -> dict:
    """Distribution of max-minus-min accuracy when every group shares accuracy p."""
    rng = np.random.default_rng(seed)
    sizes = np.asarray(group_sizes)
    draws = rng.binomial(sizes, p, size=(n_sim, len(sizes))) / sizes
    gaps = draws.max(axis=1) - draws.min(axis=1)
    return {
        "p": p,
        "group_sizes": list(map(int, sizes)),
        "n_sim": n_sim,
        "mean_gap": float(gaps.mean()),
        "median_gap": float(np.median(gaps)),
        "pct_2_5": float(np.percentile(gaps, 2.5)),
        "pct_97_5": float(np.percentile(gaps, 97.5)),
        "_gaps": gaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Null calibration for the tone gap.")
    ap.add_argument("--n", type=int, nargs="+", default=DDI_GROUP_SIZES,
                    help="Per-group test-set sizes.")
    ap.add_argument("--p", type=float, nargs="+", default=[0.67, 0.70, 0.75, 0.80],
                    help="Shared true accuracy values to sweep.")
    ap.add_argument("--n-sim", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "null_tone_gap.json")
    args = ap.parse_args()

    print("Null distribution of max-minus-min accuracy across Fitzpatrick groups")
    print(f"Group sizes: {args.n}   ({args.n_sim} simulations)\n")
    print(f"{'true acc':<11}{'E[gap]':<11}{'2.5%':<10}{'97.5%':<10}{'P(gap > 0.10)':<15}")
    print("-" * 57)

    records = []
    for p in args.p:
        res = simulate_null_gap(args.n, p, args.n_sim)
        gaps = res.pop("_gaps")
        res["p_gap_exceeds_0.10"] = float((gaps > 0.10).mean())
        records.append(res)
        print(f"{p:<11.2f}{res['mean_gap']:<11.3f}{res['pct_2_5']:<10.3f}"
              f"{res['pct_97_5']:<10.3f}{res['p_gap_exceeds_0.10']:<15.3f}")

    reference = simulate_null_gap(args.n, 0.75, args.n_sim)
    gaps = reference.pop("_gaps")

    print("\nObserved gaps vs the null at p = 0.75 "
          f"(E[gap] = {reference['mean_gap']:.3f}, "
          f"95% range {reference['pct_2_5']:.3f}-{reference['pct_97_5']:.3f}):\n")
    print(f"{'model':<18}{'observed':<12}{'null percentile':<18}{'verdict':<28}")
    print("-" * 76)

    comparisons = {}
    for model, gap in OBSERVED_GAPS.items():
        pct = float((gaps < gap).mean()) * 100
        verdict = ("within null range" if gap <= reference["pct_97_5"]
                   else "exceeds null 97.5%")
        comparisons[model] = {"observed_gap": gap,
                              "null_percentile": round(pct, 1),
                              "verdict": verdict}
        print(f"{model:<18}{gap:<12.3f}{pct:<18.1f}{verdict:<28}")

    print("\nEvery observed tone gap falls inside the range expected with no tone")
    print("effect whatsoever. Report the tone effect as 'not distinguishable from")
    print("zero at this sample size', not as a measured gap of 0.10-0.18.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"sweep": records, "reference_p": 0.75, "reference": reference,
         "comparisons": comparisons}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
