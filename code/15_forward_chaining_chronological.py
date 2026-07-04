"""Rolling-origin (forward-chaining) chronological evaluation (addresses D-06/C-08).

The main chronological result uses a single 80/20 time cut. This script sorts
records by Generate Time and evaluates four expanding-window folds: train on the
earliest 60/70/80/90 percent and test on the immediately following 10 percent.
It reports core macro-F1, balanced accuracy, and errors per fold with the
median and range, so the temporal boundary is not read off a single cutpoint.
Model and preprocessing match the selected 300-tree XGBoost pipeline.

Writes results/q1_audit_revision/forward_chaining_chronological.csv.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
CORE = ["Threat/Content Type","Application","Source Zone","Destination Zone","Inbound Interface",
        "Outbound Interface","IP Protocol","Source Port","Destination Port","Source Country",
        "Destination Country","Threat/Content Name","Category","Severity","Direction",
        "Subcategory of app","Category of app","Technology of app","Risk of app","SaaS of app"]


def build_pipe():
    num = [c for c in ["Source Port","Destination Port","Risk of app"] if c in CORE]
    cat = [c for c in CORE if c not in num]
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                          ("enc",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]), cat),
        ("num", Pipeline([("imp",SimpleImputer(strategy="median"))]), num)], remainder="drop")
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
                      colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss",
                      tree_method="hist", random_state=SEED, n_jobs=-1)
    return Pipeline([("pre",pre),("model",m)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the controlled processed CSV.")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit("Controlled processed CSV not found. Pass --data <path>.")
    print(f"Using dataset: {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    df["target"] = df["target"].astype(str)
    df["_gt"] = pd.to_datetime(df["Generate Time"], errors="coerce")
    order = df.sort_values("_gt").index.to_numpy()
    n = len(order)
    le = LabelEncoder().fit(df["target"])  # all five classes, consistent encoding
    y_all = le.transform(df["target"])

    rows = []
    for train_pct in (60, 70, 80, 90):
        tr_end = int(n * train_pct/100)
        te_end = int(n * (train_pct+10)/100)
        tr_idx = order[:tr_end]
        te_idx = order[tr_end:te_end] if te_end > tr_end else order[tr_end:]
        pipe = build_pipe()
        pipe.fit(df.loc[tr_idx, CORE], y_all[tr_idx])
        pred = pipe.predict(df.loc[te_idx, CORE])
        yt = y_all[te_idx]
        f = f1_score(yt, pred, average="macro"); b = balanced_accuracy_score(yt, pred)
        e = int((pred != yt).sum())
        rows.append({"train_pct": train_pct, "test_window": f"{train_pct}-{train_pct+10}",
                     "train_rows": len(tr_idx), "test_rows": len(te_idx),
                     "macro_f1": f, "balanced_accuracy": b, "errors": e})
        print(f"train<= {train_pct}%  test {train_pct}-{train_pct+10}%  "
              f"macroF1={f:.4f} BAcc={b:.4f} err={e} (test n={len(te_idx)})", flush=True)

    out = pd.DataFrame(rows)
    arr = out["macro_f1"].to_numpy()
    print(f"\nforward-chaining core macro-F1: median={np.median(arr):.4f} "
          f"range={arr.min():.4f}-{arr.max():.4f} mean={arr.mean():.4f}+/-{arr.std():.4f}")
    outdir = REPO_ROOT / "results/q1_audit_revision"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "forward_chaining_chronological.csv", index=False)
    print(f"Wrote: {outdir / 'forward_chaining_chronological.csv'}")


if __name__ == "__main__":
    main()
