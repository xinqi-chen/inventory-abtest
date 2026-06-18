#!/usr/bin/env python3
"""Download and validate the FreshRetailNet-50K dataset used by this project."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"
DEFAULT_REVISION = "2acbe01460f63fb293f090b6cdaf94ac588ab85c"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "frn_50k_local_dataset"

EXPECTED_ROWS = {
    "train": 4_500_000,
    "eval": 350_000,
}

REQUIRED_COLUMNS = {
    "city_id",
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FreshRetailNet-50K from Hugging Face and save it in the "
            "local Hugging Face DatasetDict format expected by the trace-driven code."
        )
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Hugging Face dataset id. Default: {DEFAULT_DATASET_ID}",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=(
            "Dataset git revision to download. The default pins the snapshot used "
            "when preparing this repository."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local save_to_disk folder. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only validate an existing saved dataset folder; do not download.",
    )
    return parser.parse_args()


def load_dataset_library():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install the trace-driven requirements first:\n"
            "  python -m pip install -r trace_driven_freshretailnet/requirements.txt"
        ) from exc
    return load_dataset


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_saved_dataset(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    dataset_dict_path = output_dir / "dataset_dict.json"
    if not dataset_dict_path.exists():
        raise SystemExit(f"Missing {dataset_dict_path}")

    dataset_dict = read_json(dataset_dict_path)
    splits = dataset_dict.get("splits", [])
    missing_splits = sorted(set(EXPECTED_ROWS) - set(splits))
    if missing_splits:
        raise SystemExit(f"Missing required split(s): {', '.join(missing_splits)}")

    summary = []
    for split, expected_rows in EXPECTED_ROWS.items():
        split_dir = output_dir / split
        info_path = split_dir / "dataset_info.json"
        state_path = split_dir / "state.json"
        if not info_path.exists():
            raise SystemExit(f"Missing {info_path}")
        if not state_path.exists():
            raise SystemExit(f"Missing {state_path}")

        info = read_json(info_path)
        state = read_json(state_path)
        split_info = info.get("splits", {}).get(split, {})
        observed_rows = split_info.get("num_examples")
        if observed_rows != expected_rows:
            raise SystemExit(
                f"Unexpected {split} row count: {observed_rows}; expected {expected_rows}"
            )

        features = set(info.get("features", {}))
        missing_columns = sorted(REQUIRED_COLUMNS - features)
        if missing_columns:
            raise SystemExit(
                f"{split} is missing required column(s): {', '.join(missing_columns)}"
            )

        data_files = state.get("_data_files", [])
        if not data_files:
            raise SystemExit(f"{split} has no Arrow data files in {state_path}")
        for data_file in data_files:
            arrow_path = split_dir / data_file.get("filename", "")
            if not arrow_path.exists():
                raise SystemExit(f"Missing Arrow shard: {arrow_path}")

        summary.append((split, observed_rows, len(data_files)))

    print(f"Verified FreshRetailNet-50K dataset at {output_dir}")
    for split, rows, shard_count in summary:
        print(f"  {split}: {rows:,} rows, {shard_count} Arrow shard(s)")


def download_dataset(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.force:
            print(f"{output_dir} already exists; validating it without re-downloading.")
            validate_saved_dataset(output_dir)
            return
        shutil.rmtree(output_dir)

    load_dataset = load_dataset_library()
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir else None

    print(f"Downloading {args.dataset_id} at revision {args.revision}")
    dataset = load_dataset(
        args.dataset_id,
        revision=args.revision,
        cache_dir=cache_dir,
    )
    print(f"Saving dataset to {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    validate_saved_dataset(output_dir)


def main() -> None:
    args = parse_args()
    if args.verify_only:
        validate_saved_dataset(args.output_dir)
    else:
        download_dataset(args)


if __name__ == "__main__":
    main()
