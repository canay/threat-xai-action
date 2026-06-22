#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
start_epoch=$(date +%s)
start_iso=$(date -Iseconds)

python3 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r shared/env/requirements-ml.txt

python shared/scripts/03_benchmark_baseline.py \
  --dataset threat \
  --feature-set core \
  --models "Decision Tree,Random Forest,Extra Trees,XGBoost,LightGBM,CatBoost" \
  --output-dir "data/reports"

end_epoch=$(date +%s)
end_iso=$(date -Iseconds)
echo "paper1 vps all baseline started=${start_iso} ended=${end_iso} wall_seconds=$((end_epoch-start_epoch))" | tee "data/reports/run_vps_all_baseline_timing.txt"
