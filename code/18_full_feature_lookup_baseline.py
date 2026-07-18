"""Full-feature exact-signature majority lookup baseline (fair non-learned comparator).

Reviewer request (ARS review panel, devil's advocate CRITICAL): the manuscript compares
the learned model only against 4-field and 10-field context-majority lookups. Because
~85.44% of records share an exact 20-feature signature, a *full-feature* exact-match
majority lookup is the fair non-learned baseline that isolates how much the learned model
adds over memorization. This script reuses the exact same evaluation logic, split, seed,
and metric definitions as code/11_q1_audit_revision_checks.py (the script that produced the
published simple_context_baselines.csv), adding only the full-core (20-feature) and
full-no-descriptor (17-feature) signature-majority baselines, and re-running the published
10-field and 4-field lookups as reproduction controls.

Aggregate-only outputs; no event-level records are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

# --- Verbatim from code/11_q1_audit_revision_checks.py -----------------------------------
CORE_FEATURES = [
    "Threat/Content Type", "Application", "Source Zone", "Destination Zone",
    "Inbound Interface", "Outbound Interface", "IP Protocol", "Source Port",
    "Destination Port", "Source Country", "Destination Country", "Threat/Content Name",
    "Category", "Severity", "Direction", "Subcategory of app", "Category of app",
    "Technology of app", "Risk of app", "SaaS of app",
]
DIRECT_THREAT_DESCRIPTORS = {"Threat/Content Name", "Threat/Content Type", "Severity"}
NO_DESC_FEATURES = [f for f in CORE_FEATURES if f not in DIRECT_THREAT_DESCRIPTORS]
MINIMAL_CONTEXT = [
    "Application", "Source Zone", "Destination Zone", "Inbound Interface",
    "Outbound Interface", "IP Protocol", "Source Port", "Destination Port",
    "Source Country", "Destination Country", "Direction",
]
# Reproduction controls: exactly the published CONTEXT_BASELINES lookups.
APP_DIR_ZONES = ["Application", "Direction", "Source Zone", "Destination Zone"]
OPERATIONAL_MINIMAL = [
    "Application", "Direction", "Source Zone", "Destination Zone",
    "Inbound Interface", "Outbound Interface", "IP Protocol", "Destination Port",
    "Source Country", "Destination Country",
]

BASELINES = {
    "context_majority_full_core": CORE_FEATURES,                 # 20 features (fair lookup)
    "context_majority_full_no_threat_descriptors": NO_DESC_FEATURES,  # 17 features
    "context_majority_full_minimal_context": MINIMAL_CONTEXT,    # 11 features
    "context_majority_operational_minimal": OPERATIONAL_MINIMAL,  # 10 (published control)
    "context_majority_app_dir_zones": APP_DIR_ZONES,             # 4 (published control)
    "train_majority": [],
}


def make_context_keys(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, ...]]:
    if not columns:
        return []
    values = frame[columns].fillna("__MISSING__").astype(str)
    return list(values.itertuples(index=False, name=None))


def majority_label(labels: pd.Series) -> str:
    counts = labels.astype(str).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    return sorted(top)[0]


def fixed_label_order(series: pd.Series) -> list[str]:
    preferred = ["Allow", "Block", "Drop", "Reset-Both", "Reset-Server"]
    observed = set(series.astype(str))
    ordered = [label for label in preferred if label in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def evaluate_string_predictions(y_true, y_pred, labels) -> dict:
    true = y_true.astype(str).to_numpy()
    pred = np.asarray(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy_fixed_labels": float(recall_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "errors": int((true != pred).sum()),
    }


def chronological_indices(df: pd.DataFrame):
    ordered = df.sort_values(["Generate Time Parsed"], kind="mergesort").index.to_numpy()
    nominal_cut = int(len(ordered) * 0.8)
    boundary_time = df.loc[ordered[nominal_cut], "Generate Time Parsed"]
    boundary_positions = np.flatnonzero(
        df.loc[ordered, "Generate Time Parsed"].to_numpy() == boundary_time.to_datetime64()
    )
    candidates = [int(boundary_positions[0]), int(boundary_positions[-1] + 1)]
    candidates = [candidate for candidate in candidates if 0 < candidate < len(ordered)]
    cut = min(candidates, key=lambda candidate: (abs(candidate - nominal_cut), candidate))
    return ordered[:cut], ordered[cut:]


def evaluate_context_baseline(df, train_idx, test_idx, split_name, baseline_name, context_columns, labels) -> dict:
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    global_label = majority_label(train["target"])
    if not context_columns:
        predictions = [global_label] * len(test)
        matched_fraction = 0.0
        mapping_size = 0
    else:
        mapping: dict[tuple[str, ...], str] = {}
        grouped = train.assign(_key=make_context_keys(train, context_columns)).groupby("_key")["target"]
        for key, group in grouped:
            mapping[key] = majority_label(group)
        test_keys = make_context_keys(test, context_columns)
        matched = [key in mapping for key in test_keys]
        predictions = [mapping.get(key, global_label) for key in test_keys]
        matched_fraction = float(np.mean(matched))
        mapping_size = len(mapping)
    metrics = evaluate_string_predictions(test["target"], predictions, labels)
    return {
        "split": split_name,
        "baseline": baseline_name,
        "n_features": len(context_columns),
        "context_columns": ";".join(context_columns) if context_columns else "(none)",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mapping_size": int(mapping_size),
        "matched_context_fraction": matched_fraction,
        **metrics,
    }
# -----------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/processed/threat_five_class.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("results/q1_audit_revision"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    df["Generate Time Parsed"] = pd.to_datetime(df["Generate Time"], errors="coerce")
    df = df[df["Generate Time Parsed"].notna()].copy().reset_index(drop=True)
    labels = fixed_label_order(df["target"])

    strat_train, strat_test = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=df["target"], random_state=args.seed
    )
    chrono_train, chrono_test = chronological_indices(df)
    splits = {
        "stratified_holdout": (strat_train, strat_test),
        "chronological_holdout": (chrono_train, chrono_test),
    }

    rows = []
    for split_name, (tr, te) in splits.items():
        for name, cols in BASELINES.items():
            rows.append(evaluate_context_baseline(df, tr, te, split_name, name, cols, labels))
    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / "full_feature_lookup_baseline.csv", index=False)

    meta = {
        "seed": args.seed,
        "n_records": int(len(df)),
        "note": "Full-feature exact-signature majority lookup vs published context lookups; "
                "verbatim reuse of code/11 evaluation logic. Aggregate-only.",
    }
    (args.outdir / "full_feature_lookup_baseline_metadata.json").write_text(json.dumps(meta, indent=2))
    print(out[["split", "baseline", "n_features", "mapping_size", "matched_context_fraction",
               "macro_f1", "balanced_accuracy_fixed_labels", "errors"]].to_string(index=False))


if __name__ == "__main__":
    main()
