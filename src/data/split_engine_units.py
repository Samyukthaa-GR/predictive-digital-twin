from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_by_engine(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    id_column = config["columns"]["id"]
    validation_size = float(config["split"]["validation_size"])
    seed = int(config["split"]["random_seed"])

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    engine_ids = np.array(sorted(frame[id_column].unique()))
    if len(engine_ids) < 2:
        raise ValueError("At least two engines are required for an engine-wise split")

    rng = np.random.default_rng(seed)
    shuffled = engine_ids.copy()
    rng.shuffle(shuffled)

    validation_count = max(1, int(round(len(shuffled) * validation_size)))
    validation_ids = np.sort(shuffled[:validation_count])
    train_ids = np.sort(shuffled[validation_count:])

    if len(train_ids) == 0:
        raise ValueError("Engine-wise split produced an empty training set")

    overlap = set(train_ids).intersection(set(validation_ids))
    if overlap:
        raise RuntimeError(f"Leakage detected: engines in both splits: {sorted(overlap)}")

    train = frame[frame[id_column].isin(train_ids)].copy()
    validation = frame[frame[id_column].isin(validation_ids)].copy()

    manifest = {
        "split_type": "engine_wise",
        "dataset_name": config["dataset"]["name"],
        "random_seed": seed,
        "rul_cap": config["labels"].get("cap"),
        "validation_size": validation_size,
        "train_engine_count": int(len(train_ids)),
        "validation_engine_count": int(len(validation_ids)),
        "train_row_count": int(len(train)),
        "validation_row_count": int(len(validation)),
        "train_engine_ids": train_ids.astype(int).tolist(),
        "validation_engine_ids": validation_ids.astype(int).tolist(),
    }
    return train, validation, manifest


def save_split_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
