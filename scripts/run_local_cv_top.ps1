$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root
$Start = Get-Date

.\.venv-ml\Scripts\python.exe shared\scripts\03_benchmark_baseline.py `
  --dataset threat `
  --feature-set core `
  --models "Extra Trees,XGBoost,LightGBM,CatBoost" `
  --cv `
  --output-dir "paper1_threat_xai_action\data\reports"

$End = Get-Date
$Elapsed = $End - $Start
"paper1 local cv top started=$($Start.ToString('s')) ended=$($End.ToString('s')) wall_seconds=$([int]$Elapsed.TotalSeconds)" |
  Tee-Object -FilePath "paper1_threat_xai_action\data\reports\run_local_cv_top_timing.txt"
