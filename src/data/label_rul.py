from __future__ import annotations

import pandas as pd


def _apply_rul_cap(
    frame: pd.DataFrame, raw_target_column: str, capped_target_column: str, cap: int | None
) -> pd.DataFrame:
    labeled = frame.copy()
    if cap is None:
        labeled[capped_target_column] = labeled[raw_target_column]
        return labeled
    labeled[capped_target_column] = labeled[raw_target_column].clip(upper=cap)
    return labeled


def label_train_rul(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    id_column = config["columns"]["id"]
    cycle_column = config["columns"]["cycle"]
    raw_target_column = config["labels"]["raw_target"]
    capped_target_column = config["labels"]["capped_target"]
    cap = config["labels"].get("cap")

    labeled = frame.copy()
    max_cycle = labeled.groupby(id_column)[cycle_column].transform("max")
    labeled[raw_target_column] = max_cycle - labeled[cycle_column]
    return _apply_rul_cap(labeled, raw_target_column, capped_target_column, cap)


def label_test_rul_for_evaluation(
    frame: pd.DataFrame, final_rul: pd.DataFrame, config: dict
) -> pd.DataFrame:
    id_column = config["columns"]["id"]
    cycle_column = config["columns"]["cycle"]
    raw_target_column = config["labels"]["raw_target"]
    capped_target_column = config["labels"]["capped_target"]
    cap = config["labels"].get("cap")

    engine_ids = sorted(frame[id_column].unique())
    if len(engine_ids) != len(final_rul):
        raise ValueError("Number of test engines must match number of provided final RUL rows")

    final_rul_by_engine = dict(zip(engine_ids, final_rul["final_rul"].to_numpy(), strict=True))
    max_observed_cycle = frame.groupby(id_column)[cycle_column].transform("max")

    labeled = frame.copy()
    # Test RUL labels are for offline evaluation only; feature construction must not use them.
    labeled[raw_target_column] = (
        max_observed_cycle - labeled[cycle_column] + labeled[id_column].map(final_rul_by_engine)
    )
    return _apply_rul_cap(labeled, raw_target_column, capped_target_column, cap)


def split_test_features_and_labels(
    labeled_test: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_column = config["columns"]["id"]
    cycle_column = config["columns"]["cycle"]
    raw_target_column = config["labels"]["raw_target"]
    capped_target_column = config["labels"]["capped_target"]

    label_columns = [id_column, cycle_column, raw_target_column, capped_target_column]
    labels = labeled_test[label_columns].copy()
    features = labeled_test.drop(columns=[raw_target_column, capped_target_column]).copy()
    return features, labels
