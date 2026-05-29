$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root
$Start = Get-Date

.\.venv-ml\Scripts\python.exe shared\scripts\03_benchmark_baseline.py `
  --dataset threat `
  --feature-set core `
  --models "Decision Tree,Random Forest,Extra Trees,XGBoost,LightGBM,CatBoost" `
  --output-dir "paper1_threat_xai_action\data\reports"

$End = Get-Date
$Elapsed = $End - $Start
"paper1 local baseline started=$($Start.ToString('s')) ended=$($End.ToString('s')) wall_seconds=$([int]$Elapsed.TotalSeconds)" |
  Tee-Object -FilePath "paper1_threat_xai_action\data\reports\run_local_baseline_timing.txt"
