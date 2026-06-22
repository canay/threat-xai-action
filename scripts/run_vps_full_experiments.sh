#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
start_epoch=$(date +%s)
start_iso=$(date -Iseconds)
outdir="data/reports"
mkdir -p "$outdir"

source .venv-ml/bin/activate 2>/dev/null || {
  python3 -m venv .venv-ml
  source .venv-ml/bin/activate
}
python -m pip install --upgrade pip
python -m pip install -r shared/env/requirements-ml.txt

common_models="Decision Tree,Random Forest,Extra Trees,XGBoost,LightGBM,CatBoost"
top_models="Extra Trees,XGBoost,LightGBM,CatBoost"

echo "Paper 1 baseline core"
python shared/scripts/03_benchmark_baseline.py \
  --dataset threat \
  --feature-set core \
  --models "$common_models" \
  --tag baseline \
  --output-dir "$outdir"

echo "Paper 1 with policy field"
python shared/scripts/03_benchmark_baseline.py \
  --dataset threat \
  --feature-set with_policy \
  --models "$common_models" \
  --tag with_policy \
  --output-dir "$outdir"

echo "Paper 1 ablation without high-leakage threat descriptors"
python shared/scripts/03_benchmark_baseline.py \
  --dataset threat \
  --feature-set core \
  --models "$common_models" \
  --exclude-features "Threat/Content Name,Threat/Content Type,Severity" \
  --tag no_threat_descriptors \
  --output-dir "$outdir"

echo "Paper 1 top-model CV core"
python shared/scripts/03_benchmark_baseline.py \
  --dataset threat \
  --feature-set core \
  --models "$top_models" \
  --cv \
  --tag top_cv \
  --output-dir "$outdir"

end_epoch=$(date +%s)
end_iso=$(date -Iseconds)
echo "paper1 vps full experiments started=${start_iso} ended=${end_iso} wall_seconds=$((end_epoch-start_epoch))" | tee "$outdir/run_vps_full_experiments_timing.txt"
