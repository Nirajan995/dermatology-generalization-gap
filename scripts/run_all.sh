#!/usr/bin/env bash
# End-to-end reproduction. Assumes the four dataset roots are set (see docs/DATA.md)
# and that `pip install -e ".[full]"` has been run.
#
#   bash scripts/run_all.sh
#
# Stages 2 and 3 need a GPU and are individually resumable; stages 1, 4 and 5 are
# CPU-only. Re-running is safe: completed work is skipped.

set -euo pipefail

ARTIFACTS="${DERMGAP_ARTIFACTS:-./artifacts}"
RESULTS="${DERMGAP_RESULTS:-./results}"

echo "=== 1/5  Building unified dataset tables ==============================="
python -m dermgap.datasets

echo "=== 2/5  Fine-tuning the cancer baseline (GPU) ========================="
if [ -f "${ARTIFACTS}/checkpoints/baseline_resnet50_best.pt" ]; then
  echo "    checkpoint already present, skipping"
else
  python -m dermgap.train_baseline --epochs 15
fi

echo "=== 3/5  Extracting frozen features (GPU, resumable) ==================="
echo "    DINOv3 is gated: run 'huggingface-cli login' first if this fails."
python -m dermgap.extract_features --models all --datasets all

echo "=== 4/5  Running all three analyses (CPU) =============================="
python -m dermgap.evaluate --analysis all

echo "=== 5/5  Figures and tone-gap null calibration (CPU) ==================="
python -m dermgap.figures --granularity fine
python -m dermgap.figures --granularity category
python -m dermgap.null_calibration

echo
echo "Done. Results in ${RESULTS}/ — see summary.md."
echo "Compare against ${RESULTS}/original_results.json to verify the reproduction."
