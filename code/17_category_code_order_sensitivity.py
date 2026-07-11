"""Category-code-order sensitivity for the selected XGBoost probe.

The manuscript uses deterministic ordinal codes for nominal categorical fields
inside each training partition. This script tests whether the reported
stratified selected-model results depend on that arbitrary code order by
holding the train/test split and XGBoost configuration fixed, then refitting the
model after random permutations of every categorical feature's training
category-to-code map. Missing categorical values are imputed with the
training-partition mode before either the reference or permuted map is fitted,
matching the canonical selected-model preprocessing contract.

The event-level CSV is controlled-access and must be passed explicitly with
--data. Outputs are aggregate metrics only; no event-level records are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

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

NUMERIC_COLUMN_ORDER = ["Source Port", "Destination Port", "Risk of app"]
NUMERIC_COLUMNS = set(NUMERIC_COLUMN_ORDER)


def resolve_data(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_file():
        return candidate
    raise SystemExit("Controlled processed CSV not found. Pass --data <path>.")


def xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def fit_numeric(train: pd.DataFrame, features: list[str]) -> dict[str, float]:
    medians = {}
    for col in features:
        if col in NUMERIC_COLUMNS:
            medians[col] = float(pd.to_numeric(train[col], errors="coerce").median())
    return medians


def fit_category_maps(
    train: pd.DataFrame,
    features: list[str],
    *,
    variant: str,
    rng: np.random.Generator | None,
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    maps: dict[str, dict[str, int]] = {}
    fill_values: dict[str, str] = {}
    for col in features:
        if col in NUMERIC_COLUMNS:
            continue
        series = train[col].astype("string")
        mode = series.mode(dropna=True)
        if mode.empty:
            raise ValueError(f"Categorical feature has no observed training value: {col}")
        fill_values[col] = str(mode.iloc[0])
        imputed = series.fillna(fill_values[col])
        categories = sorted(imputed.unique().tolist())
        codes = np.arange(len(categories), dtype=int)
        if variant == "permuted":
            if rng is None:
                raise ValueError("rng is required for permuted variant")
            codes = rng.permutation(codes)
        maps[col] = {str(cat): int(code) for cat, code in zip(categories, codes)}
    return maps, fill_values


def transform(
    frame: pd.DataFrame,
    features: list[str],
    *,
    category_maps: dict[str, dict[str, int]],
    fill_values: dict[str, str],
    numeric_medians: dict[str, float],
) -> np.ndarray:
    columns = []
    # Match the canonical ColumnTransformer exactly: all categorical columns
    # in manuscript feature order, followed by numeric columns in the fixed
    # Source Port / Destination Port / Risk of app order.
    categorical_order = [col for col in features if col not in NUMERIC_COLUMNS]
    numeric_order = [col for col in NUMERIC_COLUMN_ORDER if col in features]
    for col in categorical_order + numeric_order:
        if col in NUMERIC_COLUMNS:
            values = pd.to_numeric(frame[col], errors="coerce").fillna(numeric_medians[col]).to_numpy(dtype=float)
        else:
            values = (
                frame[col]
                .astype("string")
                .fillna(fill_values[col])
                .map(category_maps[col])
                .fillna(-1)
                .to_numpy(dtype=float)
            )
        columns.append(values)
    return np.column_stack(columns)


def evaluate_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    *,
    variant_name: str,
    map_variant: str,
    model_seed: int,
    map_seed: int | None,
) -> dict[str, float | int | str | None]:
    rng = np.random.default_rng(map_seed) if map_seed is not None else None
    numeric_medians = fit_numeric(train, features)
    category_maps, fill_values = fit_category_maps(train, features, variant=map_variant, rng=rng)
    x_train = transform(
        train,
        features,
        category_maps=category_maps,
        fill_values=fill_values,
        numeric_medians=numeric_medians,
    )
    x_test = transform(
        test,
        features,
        category_maps=category_maps,
        fill_values=fill_values,
        numeric_medians=numeric_medians,
    )
    model = xgb(model_seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    per_class = f1_score(y_test, pred, average=None, labels=list(range(len(classes))), zero_division=0)
    row: dict[str, float | int | str | None] = {
        "variant": variant_name,
        "map_seed": map_seed,
        "macro_f1": float(f1_score(y_test, pred, average="macro", labels=list(range(len(classes))), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "errors": int((pred != y_test).sum()),
    }
    for label, value in zip(classes, per_class):
        row[f"f1_{label.lower().replace('-', '_')}"] = float(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "results/q1_audit_revision")
    parser.add_argument("--seed", type=int, default=42, help="Train/test split and model seed.")
    parser.add_argument("--map-seeds", type=int, nargs="+", default=list(range(20260708, 20260713)))
    args = parser.parse_args()

    data_path = resolve_data(args.data)
    print(f"Using dataset: {data_path}")
    df = pd.read_csv(data_path)
    encoder = LabelEncoder().fit(df["target"].astype(str))
    classes = list(encoder.classes_)
    train, test = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=args.seed)
    y_train = encoder.transform(train["target"].astype(str))
    y_test = encoder.transform(test["target"].astype(str))

    rows = []
    for feature_set, features in FEATURE_SETS.items():
        print(f"\nFeature set: {feature_set}")
        reference = evaluate_variant(
            train,
            test,
            features,
            y_train,
            y_test,
            classes,
            variant_name="ordinal_reference",
            map_variant="ordinal",
            model_seed=args.seed,
            map_seed=None,
        )
        reference["feature_set"] = feature_set
        rows.append(reference)
        print(
            f"  ordinal_reference macroF1={reference['macro_f1']:.6f} "
            f"BAcc={reference['balanced_accuracy']:.6f} err={reference['errors']}",
            flush=True,
        )
        for map_seed in args.map_seeds:
            row = evaluate_variant(
                train,
                test,
                features,
                y_train,
                y_test,
                classes,
                variant_name="permuted_category_codes",
                map_variant="permuted",
                model_seed=args.seed,
                map_seed=map_seed,
            )
            row["feature_set"] = feature_set
            rows.append(row)
            print(
                f"  permuted seed={map_seed} macroF1={row['macro_f1']:.6f} "
                f"BAcc={row['balanced_accuracy']:.6f} err={row['errors']}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    detail_path = args.outdir / "category_code_order_sensitivity.csv"
    summary_path = args.outdir / "category_code_order_sensitivity_summary.csv"
    metadata_path = args.outdir / "category_code_order_sensitivity_metadata.json"
    results.to_csv(detail_path, index=False)

    summary_rows = []
    for feature_set, group in results.groupby("feature_set"):
        reference = group[group["variant"] == "ordinal_reference"].iloc[0]
        permuted = group[group["variant"] == "permuted_category_codes"]
        summary_rows.append({
            "feature_set": feature_set,
            "ordinal_macro_f1": reference["macro_f1"],
            "permuted_macro_f1_mean": permuted["macro_f1"].mean(),
            "permuted_macro_f1_std": permuted["macro_f1"].std(ddof=1),
            "permuted_macro_f1_min": permuted["macro_f1"].min(),
            "permuted_macro_f1_max": permuted["macro_f1"].max(),
            "max_abs_macro_f1_delta_vs_ordinal": (permuted["macro_f1"] - reference["macro_f1"]).abs().max(),
            "ordinal_balanced_accuracy": reference["balanced_accuracy"],
            "permuted_balanced_accuracy_mean": permuted["balanced_accuracy"].mean(),
            "permuted_balanced_accuracy_min": permuted["balanced_accuracy"].min(),
            "permuted_balanced_accuracy_max": permuted["balanced_accuracy"].max(),
            "n_permutations": len(permuted),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "script": "code/17_category_code_order_sensitivity.py",
        "data_identifier": data_path.name,
        "data_sha256": sha256(data_path),
        "rows": int(len(df)),
        "seed": args.seed,
        "map_seeds": args.map_seeds,
        "preprocessing_contract": {
            "categorical_imputation": "training-partition most frequent value",
            "numeric_imputation": "training-partition median",
            "unknown_category_code": -1,
            "transformed_feature_order": "categorical features in manuscript order, then Source Port, Destination Port, Risk of app",
            "controlled_factor": "training category-to-code order only",
        },
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "model": {
            "family": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
        },
        "outputs": [detail_path.name, summary_path.name],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== CATEGORY CODE-ORDER SENSITIVITY SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nWrote: {detail_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {metadata_path}")


if __name__ == "__main__":
    main()
