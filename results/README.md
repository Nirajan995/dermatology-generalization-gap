# Results

## Tracked in git

| File | What it is |
|---|---|
| `original_results.json` | Every number the original Kaggle notebook runs produced, with provenance and caveats. **The reference for checking a reproduction.** |
| `purity_reference.json` | The Figure 2 purity numbers in the schema `dermgap.figures` reads, transcribed from `original_results.json`. |
| `null_tone_gap.json` | Null distribution of the DDI tone gap at the observed group sizes. See §2 of `docs/REVIEW_NOTES.md`. |

## Generated (gitignored)

Produced by `python -m dermgap.evaluate --analysis all`:

| File | Corresponds to |
|---|---|
| `decomposition.json` | Table 2 — in-domain vs SCIN vs DDI tone gap |
| `purity.json` | Figure 2 — label-free kNN neighbour-purity lift |
| `adaptation.json` | Table 3 — full probe vs few-shot vs raw features |
| `summary.md` | All three as markdown tables |

## Verifying a reproduction

```bash
python -m dermgap.evaluate --analysis all
diff <(python -c "import json;print(json.dumps(json.load(open('results/decomposition.json')),indent=2,sort_keys=True))") -
```

Then compare `summary.md` against the `table2_decomposition`, `figure2_knn_purity` and `table3_adaptation` blocks of `original_results.json`. Values should agree to the precision reported in the paper.

Two known sources of small divergence, both documented in `original_results.json`:

1. **Point estimate vs bootstrap mean.** The paper reports point estimates for probe accuracy. Bootstrap means differ (e.g. SCIN category, cancer baseline: 0.207 point vs 0.172 bootstrap mean) because resampling sometimes drops a rare category and changes the per-class average. `evaluate.py` reports both.
2. **SCIN category purity.** The original Figure-2 category numbers came from an earlier mapping that omitted the `traumatic_other` and `pigmentary` groups, giving 5 categories. `dermgap.taxonomy` is now the single canonical mapping with 7 categories, so rerunning gives slightly different values. Model ordering is unchanged.
