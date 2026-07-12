"""Source-port near-duplicate leakage ablation (addresses adversarial D-04).

Exact-signature grouped splits still let records that differ only in an ephemeral
source port fall into different groups, so burst siblings can cross the
train/test boundary. This script:
  (1) characterizes Source Port (cardinality, share of ephemeral high ports);
  (2) measures how many exact core signatures merge when Source Port is dropped
      from the signature (evidence of a burst pseudo-identifier);
  (3) runs a 5-split GroupShuffleSplit whose GROUPING signature excludes Source
      Port, while the model still TRAINS on the full core features, and reports
      core macro-F1. If the score stays high, exact-signature grouping was not
      hiding a source-port burst-memorization channel.

Prints a compact report and, when ``--outdir`` is supplied, writes the same
aggregate evidence to ``sourceport_nearduplicate_ablation.csv``.
"""
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent


CORE = ["Threat/Content Type","Application","Source Zone","Destination Zone","Inbound Interface",
        "Outbound Interface","IP Protocol","Source Port","Destination Port","Source Country",
        "Destination Country","Threat/Content Name","Category","Severity","Direction",
        "Subcategory of app","Category of app","Technology of app","Risk of app","SaaS of app"]
SEED = 42

def signature(df, feats):
    v = df[feats].astype("string").fillna("<NA>")
    return v.agg("\x1f".join, axis=1).map(lambda s: hashlib.sha1(s.encode()).hexdigest())

def build_pipe(feats):
    num = [c for c in ["Source Port","Destination Port","Risk of app"] if c in feats]
    cat = [c for c in feats if c not in num]
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                          ("enc",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]), cat),
        ("num", Pipeline([("imp",SimpleImputer(strategy="median"))]), num)], remainder="drop")
    m = XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,
                      objective="multi:softprob",eval_metric="mlogloss",tree_method="hist",random_state=SEED,n_jobs=-1)
    return Pipeline([("pre",pre),("model",m)])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the controlled processed CSV.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional directory for the aggregate reproducibility CSV.",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit("Controlled processed CSV not found. Pass --data <path>.")
    df = pd.read_csv(args.data, low_memory=False); df["target"]=df["target"].astype(str)
    n=len(df)
    # (1) Source Port characterization
    sp = pd.to_numeric(df["Source Port"], errors="coerce")
    card = df["Source Port"].nunique(dropna=True)
    ephemeral = float((sp>=1024).mean())
    dynamic = float((sp>=49152).mean())
    print(f"[1] rows={n}  Source Port distinct={card}  share>=1024(ephemeral)={ephemeral:.3f}  "
          f"share>=49152(dynamic)={dynamic:.3f}")
    # (2) signature merge when dropping Source Port
    core_sig = signature(df, CORE)
    nosp_sig = signature(df, [c for c in CORE if c!="Source Port"])
    print(f"[2] unique core signatures={core_sig.nunique()}  "
          f"unique signatures w/o Source Port={nosp_sig.nunique()}  "
          f"(merge = {core_sig.nunique()-nosp_sig.nunique()} groups collapse)")
    dup_core = float(core_sig.duplicated(keep=False).mean())
    dup_nosp = float(nosp_sig.duplicated(keep=False).mean())
    print(f"    duplicate-row fraction: core-signature={dup_core:.4f}  no-source-port-signature={dup_nosp:.4f}")
    # (3) grouped split using no-source-port signature; model trains on full core
    groups = nosp_sig.to_numpy()
    y = LabelEncoder().fit_transform(df["target"])
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
    f1s, bacc, errs = [], [], []
    split_rows = []
    for i,(tr,te) in enumerate(gss.split(np.zeros(n), df["target"], groups),1):
        if not set(y[te]).issubset(set(y[tr])):
            print(f"    split {i}: skipped (missing train class)"); continue
        pipe = build_pipe(CORE)
        pipe.fit(df.iloc[tr][CORE], y[tr])
        pred = pipe.predict(df.iloc[te][CORE])
        f = f1_score(y[te],pred,average="macro"); b=balanced_accuracy_score(y[te],pred); e=int((pred!=y[te]).sum())
        f1s.append(f); bacc.append(b); errs.append(e)
        split_rows.append((i, f, b, e, len(te)))
        print(f"    split {i}: macroF1={f:.4f} BAcc={b:.4f} err={e} (test n={len(te)})", flush=True)
    if f1s:
        arr=np.array(f1s)
        print(f"[3] source-port-excluded grouped split: median macroF1={np.median(arr):.4f}  "
              f"range={arr.min():.4f}-{arr.max():.4f}  mean={arr.mean():.4f}+/-{arr.std():.4f}")
        if args.outdir is not None:
            rows = [
                {"metric": "source_port_distinct", "value": card, "note": "high-cardinality"},
                {"metric": "source_port_share_ge_1024", "value": f"{ephemeral:.3f}", "note": "share of records with source port >= 1024"},
                {"metric": "source_port_share_ge_49152", "value": f"{dynamic:.3f}", "note": "share in dynamic/ephemeral range"},
                {"metric": "unique_core_signatures", "value": core_sig.nunique(), "note": "exact core feature signatures"},
                {"metric": "unique_signatures_no_source_port", "value": nosp_sig.nunique(), "note": "signatures with Source Port dropped"},
                {"metric": "signature_groups_collapsed", "value": core_sig.nunique()-nosp_sig.nunique(), "note": "groups merged when Source Port removed"},
                {"metric": "duplicate_row_fraction_core", "value": f"{dup_core:.4f}", "note": "exact core signature"},
                {"metric": "duplicate_row_fraction_no_source_port", "value": f"{dup_nosp:.4f}", "note": "signature without Source Port"},
            ]
            rows.extend(
                {"metric": f"grouped_nosp_split{i}_macro_f1", "value": f"{f:.4f}", "note": f"errors={e}"}
                for i, f, _b, e, _n in split_rows
            )
            rows.extend([
                {"metric": "grouped_nosp_median_macro_f1", "value": f"{np.median(arr):.4f}", "note": f"median over {len(arr)} splits"},
                {"metric": "grouped_nosp_range_min", "value": f"{arr.min():.4f}", "note": ""},
                {"metric": "grouped_nosp_range_max", "value": f"{arr.max():.4f}", "note": ""},
                {"metric": "grouped_nosp_mean_macro_f1", "value": f"{arr.mean():.4f}", "note": ""},
                {"metric": "grouped_nosp_std_macro_f1", "value": f"{arr.std():.4f}", "note": ""},
                {
                    "metric": "config",
                    "value": "n_estimators=300 max_depth=6 seed=42 GroupShuffleSplit test_size=0.2 n_splits=5",
                    "note": "grouping excludes Source Port; model trains on full core",
                },
            ])
            args.outdir.mkdir(parents=True, exist_ok=True)
            output_path = args.outdir / "sourceport_nearduplicate_ablation.csv"
            pd.DataFrame(rows, columns=["metric", "value", "note"]).to_csv(output_path, index=False)
            print(f"Wrote: {output_path}")

if __name__=="__main__":
    main()
