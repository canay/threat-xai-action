# regen_queue_300.py  — github-threat-xai-action/ içinde çalıştır
import argparse, importlib.util
from pathlib import Path
import pandas as pd
spec = importlib.util.spec_from_file_location("a9", "code/09_operational_review_and_context_audit.py")
a9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a9)
DATA = Path(r"..\data\processed\threat_five_class.csv")   # ana projedeki işlenmiş veri
args = argparse.Namespace(data=DATA, seed=42,
    n_estimators=300, max_depth=6, device="cpu", min_context_rows=1000, max_contexts_per_column=8)
df = pd.read_csv(args.data, low_memory=False); df["target"] = df["target"].astype(str)
_, q, _ = a9.run_review_queue(df, args)
cols = ["feature_set","scenario","queue_rows","queue_fraction","total_errors","reset_related_in_queue"]
print(q[q.scenario.isin(["low1_or_reset","mismatch_or_low5_or_reset"])][cols].to_string(index=False))