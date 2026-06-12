from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_rul import (
    label_test_rul_for_evaluation,
    label_train_rul,
    split_test_features_and_labels,
)
from src.data.load_cmapss import load_fd001_raw
from src.data.split_engine_units import save_split_manifest, split_by_engine
from src.data.validate_schema import validate_cmapss_frame, validate_rul_frame
from src.utils.config import load_config
from src.utils.reproducibility import set_global_seed


def prepare_data(config_path: str | Path) -> None:
    config = load_config(config_path)
    set_global_seed(int(config["split"]["random_seed"]))

    train_raw, test_raw, test_final_rul = load_fd001_raw(config)

    validate_cmapss_frame(train_raw, config, "FD001 train")
    validate_cmapss_frame(test_raw, config, "FD001 test")
    validate_rul_frame(
        test_final_rul,
        expected_units=test_raw[config["columns"]["id"]].nunique(),
        name="FD001 test RUL",
    )

    train_labeled_all = label_train_rul(train_raw, config)
    test_labeled = label_test_rul_for_evaluation(test_raw, test_final_rul, config)
    test_features, test_labels = split_test_features_and_labels(test_labeled, config)
    train_labeled, validation_labeled, manifest = split_by_engine(train_labeled_all, config)

    save_split_manifest(manifest, config["split"]["manifest_path"])
    _write_outputs(train_labeled, validation_labeled, test_features, test_labels, config)


def _write_outputs(train_labeled, validation_labeled, test_features, test_labels, config: dict) -> None:
    output_config = config["outputs"]
    output_dir = Path(output_config["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_labeled.to_csv(output_dir / output_config["train_labeled"], index=False)
    validation_labeled.to_csv(output_dir / output_config["validation_labeled"], index=False)
    test_features.to_csv(output_dir / output_config["test_features"], index=False)
    test_labels.to_csv(output_dir / output_config["test_labels"], index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe FD001 data artifacts.")
    parser.add_argument(
        "--config",
        default="configs/data/fd001.yaml",
        help="Path to the FD001 data configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args.config)
