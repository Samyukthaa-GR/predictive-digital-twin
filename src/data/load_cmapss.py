from __future__ import annotations

from pathlib import Path

import pandas as pd


def fd001_columns(config: dict) -> list[str]:
    columns = config["columns"]
    return [
        columns["id"],
        columns["cycle"],
        *columns["operational_settings"],
        *columns["sensors"],
    ]


def load_cmapss_table(path: str | Path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    if frame.empty:
        raise ValueError(f"C-MAPSS file is empty: {path}")
    return frame


def load_fd001_raw(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = config["dataset"]
    raw_dir = Path(dataset["raw_dir"])
    columns = fd001_columns(config)

    train = load_cmapss_table(raw_dir / dataset["train_file"], columns)
    test = load_cmapss_table(raw_dir / dataset["test_file"], columns)
    rul = pd.read_csv(raw_dir / dataset["rul_file"], sep=r"\s+", header=None, names=["final_rul"])

    if rul.empty:
        raise ValueError(f"RUL file is empty: {raw_dir / dataset['rul_file']}")

    return train, test, rul
