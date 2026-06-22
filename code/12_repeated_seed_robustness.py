"""Repeated-seed robustness check for the selected XGBoost model.

Re-runs the stratified 80/20 holdout across multiple random seeds for the core
and no-threat-descriptors feature sets, reporting macro-F1, balanced accuracy,
and per-class reset F1 as mean +/- std. This addresses single-seed sensitivity
for the rare reset classes (Reset-Both, Reset-Server). The preprocessing and
XGBoost configuration match 05_q1_validation_extensions.py exactly, so seed 42
reproduces the selected-model core result reported in the manuscript.

The event-level CSV (threat_five_class.csv) is controlled-access and is NOT in
the public repository. The script auto-locates it: it looks under the project
root (the parent of this repository), then the current directory, then the
repository. You can also pass --data explicitly. Results are written to the
repository's results/q1_audit_revision/ regardless of the current directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent      # github-threat-xai-action/
PROJECT_ROOT = REPO_ROOT.parent                          # fw1_threat_xai_action/
REL = Path("data/processed/threat_five_class.csv")

CORE_FEATURES = [
    "Threat/Content Type", "Application", "Source Zone", "Destination Zone",
    "Inbound Interface", "Outbound Interface", "IP Protocol", "Source Port",
    "Destination Port", "Source Country", "Destination Country",
    "Threat/Content Name", "Category", "Severity", "Direction",
    "Subcategory of app", "Category of app", "Technology of app",
    "Risk of app", "SaaS of app",
]
FEATURE_SETS = {
    "core": CORE_FEATURES,
    "no_threat_descriptors": [
        f for f in CORE_FEATURES
        if f not in {"Threat/Content Name", "Threat/Content Type", "Severity"}
    ],
}


def resolve_data(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates += [Path.cwd() / REL, PROJECT_ROOT / REL, REPO_ROOT / REL]
    for c in candidates:
        if c.is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        "Could not find the event-level dataset 'threat_five_class.csv'.\n"
        "It is controlled-access and not in the public repo. Tried:\n  " + tried +
        "\nPass the path explicitly, e.g.:\n"
        "  python code/12_repeated_seed_robustness.py --data /path/to/threat_five_class.csv"
    )


def build_pipeline(features, seed):
    numeric_cols = [c for c in ["Source Port", "Destination Port", "Risk of app"] if c in features]
    categorical_cols = [c for c in features if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_cols),
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ],
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, eval_metric="mlogloss", tree_method="hist",
        random_state=seed, n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=None,
                   help="Path to threat_five_class.csv (auto-located if omitted).")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results/q1_audit_revision")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    args = p.parse_args()

    data_path = resolve_data(args.data)
    print(f"Using dataset: {data_path}")
    df = pd.read_csv(data_path)
    le = LabelEncoder().fit(df["target"])
    classes = list(le.classes_)
    rb_i, rs_i = classes.index("Reset-Both"), classes.index("Reset-Server")

    rows = []
    for fs_name, feats in FEATURE_SETS.items():
        for seed in args.seeds:
            tr, te = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=seed)
            pipe = build_pipeline(feats, seed)
            pipe.fit(tr[feats], le.transform(tr["target"]))
            pred = pipe.predict(te[feats])
            y = le.transform(te["target"])
            per = f1_score(y, pred, average=None, labels=list(range(len(classes))))
            rows.append({
                "feature_set": fs_name, "seed": seed,
                "macro_f1": f1_score(y, pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "reset_both_f1": per[rb_i], "reset_server_f1": per[rs_i],
                "errors": int((pred != y).sum()),
            })
            print(f"{fs_name} seed={seed} macroF1={rows[-1]['macro_f1']:.4f} "
                  f"RB-F1={per[rb_i]:.4f} RS-F1={per[rs_i]:.4f} err={rows[-1]['errors']}", flush=True)

    res = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.outdir / "repeated_seed_results.csv", index=False)
    agg = res.groupby("feature_set").agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        reset_both_f1_mean=("reset_both_f1", "mean"), reset_both_f1_std=("reset_both_f1", "std"),
        reset_server_f1_mean=("reset_server_f1", "mean"), reset_server_f1_std=("reset_server_f1", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    agg.to_csv(args.outdir / "repeated_seed_summary.csv", index=False)
    print("\n=== SUMMARY (mean +/- std across seeds) ===")
    print(agg.to_string(index=False))
    print(f"\nWrote: {args.outdir / 'repeated_seed_results.csv'}")
    print(f"Wrote: {args.outdir / 'repeated_seed_summary.csv'}")


if __name__ == "__main__":
    main()
