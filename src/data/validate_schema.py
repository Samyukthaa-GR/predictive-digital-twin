from __future__ import annotations

import pandas as pd


def validate_cmapss_frame(frame: pd.DataFrame, config: dict, name: str) -> None:
    expected_columns = [
        config["columns"]["id"],
        config["columns"]["cycle"],
        *config["columns"]["operational_settings"],
        *config["columns"]["sensors"],
    ]

    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    extra = [column for column in frame.columns if column not in expected_columns]
    if extra:
        raise ValueError(f"{name} has unexpected columns: {extra}")

    id_column = config["columns"]["id"]
    cycle_column = config["columns"]["cycle"]

    if frame[[id_column, cycle_column]].isna().any().any():
        raise ValueError(f"{name} contains null unit or cycle values")

    duplicated = frame.duplicated([id_column, cycle_column])
    if duplicated.any():
        count = int(duplicated.sum())
        raise ValueError(f"{name} contains {count} duplicate unit-cycle rows")

    cycle_diffs = frame.groupby(id_column, sort=False)[cycle_column].diff().dropna()
    if (cycle_diffs <= 0).any():
        raise ValueError(f"{name} has non-increasing cycles within at least one engine")

    numeric_columns = expected_columns
    non_numeric = [
        column for column in numeric_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"{name} has non-numeric columns: {non_numeric}")


def validate_rul_frame(rul: pd.DataFrame, expected_units: int, name: str = "RUL") -> None:
    if list(rul.columns) != ["final_rul"]:
        raise ValueError(f"{name} must contain exactly one column named final_rul")

    if len(rul) != expected_units:
        raise ValueError(
            f"{name} row count ({len(rul)}) does not match test engine count ({expected_units})"
        )

    if rul["final_rul"].isna().any():
        raise ValueError(f"{name} contains null RUL values")

    if not pd.api.types.is_numeric_dtype(rul["final_rul"]):
        raise ValueError(f"{name} final_rul column must be numeric")

    if (rul["final_rul"] < 0).any():
        raise ValueError(f"{name} contains negative RUL values")
