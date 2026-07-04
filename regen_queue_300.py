import argparse
import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent


def load_review_module():
    module_path = REPO_ROOT / "code" / "09_operational_review_and_context_audit.py"
    spec = importlib.util.spec_from_file_location("review_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the 300-tree review-queue summary from an authorized dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to the controlled processed CSV.")
    args_cli = parser.parse_args()
    if not args_cli.data.is_file():
        raise SystemExit("Controlled processed CSV not found. Pass --data <path>.")

    review_audit = load_review_module()
    args = argparse.Namespace(
        data=args_cli.data,
        seed=42,
        n_estimators=300,
        max_depth=6,
        device="cpu",
        min_context_rows=1000,
        max_contexts_per_column=8,
    )
    df = pd.read_csv(args.data, low_memory=False)
    df["target"] = df["target"].astype(str)
    _, queue, _ = review_audit.run_review_queue(df, args)
    cols = [
        "feature_set",
        "scenario",
        "queue_rows",
        "queue_fraction",
        "total_errors",
        "reset_related_in_queue",
    ]
    print(queue[queue.scenario.isin(["low1_or_reset", "mismatch_or_low5_or_reset"])][cols].to_string(index=False))


if __name__ == "__main__":
    main()
