"""Class-weight sensitivity check for the selected XGBoost model.

The main results use an unweighted XGBoost probe. This script re-fits the same
stratified 80/20 holdout (seed 42) with inverse-frequency sample weights
w_k = n / (|C| * n_k) and reports macro-F1, balanced accuracy, and per-class
reset F1 for the weighted vs unweighted model, in the core and
no-threat-descriptors settings. It quantifies whether the unweighted choice
understates minority-class (Reset-Both / Reset-Server) performance
(reviewer concern: C7 / D9 in the round-2 audit).

Pipeline matches 05_q1_validation_extensions.py. Event-level CSV is
controlled-access and auto-located (project root, cwd, then repo).
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

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
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


def resolve_data(explicit):
    cands = ([Path(explicit)] if explicit else []) + [Path.cwd()/REL, PROJECT_ROOT/REL, REPO_ROOT/REL]
    for c in cands:
        if c.is_file():
            return c
    raise SystemExit("threat_five_class.csv not found (controlled-access). Pass --data <path>.\nTried:\n  " +
                     "\n  ".join(str(c) for c in cands))


def build_pipeline(features, seed):
    numeric = [c for c in ["Source Port", "Destination Port", "Risk of app"] if c in features]
    categ = [c for c in features if c not in numeric]
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categ),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric),
        ], verbose_feature_names_out=False)
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
                          colsample_bytree=0.9, eval_metric="mlogloss", tree_method="hist",
                          random_state=seed, n_jobs=-1)
    return pre, model


def inverse_freq_weights(y):
    n = len(y); classes, counts = np.unique(y, return_counts=True)
    w = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w[v] for v in y])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "results/q1_audit_revision")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = resolve_data(args.data); print(f"Using dataset: {data}")
    df = pd.read_csv(data)
    le = LabelEncoder().fit(df["target"]); classes = list(le.classes_)
    rb, rs = classes.index("Reset-Both"), classes.index("Reset-Server")

    rows = []
    for fs, feats in FEATURE_SETS.items():
        tr, te = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=args.seed)
        ytr, yte = le.transform(tr["target"]), le.transform(te["target"])
        for regime in ("unweighted", "inverse_frequency"):
            pre, model = build_pipeline(feats, args.seed)
            Xtr = pre.fit_transform(tr[feats]); Xte = pre.transform(te[feats])
            sw = inverse_freq_weights(ytr) if regime == "inverse_frequency" else None
            model.fit(Xtr, ytr, sample_weight=sw)
            pred = model.predict(Xte)
            per = f1_score(yte, pred, average=None, labels=list(range(len(classes))))
            rows.append({"feature_set": fs, "weighting": regime,
                         "macro_f1": f1_score(yte, pred, average="macro"),
                         "balanced_accuracy": balanced_accuracy_score(yte, pred),
                         "reset_both_f1": per[rb], "reset_server_f1": per[rs],
                         "errors": int((pred != yte).sum())})
            print(f"{fs:22s} {regime:18s} macroF1={rows[-1]['macro_f1']:.4f} "
                  f"BAcc={rows[-1]['balanced_accuracy']:.4f} RB={per[rb]:.4f} RS={per[rs]:.4f} err={rows[-1]['errors']}", flush=True)

    out = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outdir / "classweight_sensitivity.csv", index=False)
    print("\n=== CLASS-WEIGHT SENSITIVITY ===")
    print(out.to_string(index=False))
    print(f"\nWrote: {args.outdir / 'classweight_sensitivity.csv'}")


if __name__ == "__main__":
    main()
