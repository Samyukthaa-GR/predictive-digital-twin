from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSequence:
    """Ordered observations for one engine unit."""

    engine_id: Any
    rows: pd.DataFrame


class EngineSequenceIterable:
    """Iterable view over isolated, cycle-ordered engine sequences.

    This class validates a prepared FD001 dataframe at the engine boundary and
    exposes one engine trajectory at a time. It performs no feature engineering,
    statistical transformation, label transformation, or cross-engine operation.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        engine_id_column: str,
        cycle_column: str,
        target_column: str | None = None,
        copy_rows: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self._frame = frame
        self.engine_id_column = engine_id_column
        self.cycle_column = cycle_column
        self.target_column = target_column
        self.copy_rows = copy_rows
        self.logger = logger or LOGGER

        self._engine_ids = self._validate_frame()
        self.logger.info("Prepared %d isolated engine sequences", len(self._engine_ids))

    @classmethod
    def from_config(
        cls,
        frame: pd.DataFrame,
        config: dict,
        *,
        target_column: str | None = None,
        copy_rows: bool = True,
        logger: logging.Logger | None = None,
    ) -> EngineSequenceIterable:
        """Create an iterable using project FD001 column names from config."""
        columns = config["columns"]
        if target_column is None:
            target_column = config.get("labels", {}).get("capped_target")
        return cls(
            frame,
            engine_id_column=columns["id"],
            cycle_column=columns["cycle"],
            target_column=target_column,
            copy_rows=copy_rows,
            logger=logger,
        )

    def __len__(self) -> int:
        return len(self._engine_ids)

    def __iter__(self) -> Iterator[EngineSequence]:
        for engine_id in self._engine_ids:
            mask = self._frame[self.engine_id_column] == engine_id
            rows = self._frame.loc[mask]
            self._validate_engine_order(engine_id, rows)
            sequence_rows = rows.copy(deep=True) if self.copy_rows else rows
            yield EngineSequence(engine_id=engine_id, rows=sequence_rows)

    def _validate_frame(self) -> list[Any]:
        if self._frame.empty:
            raise ValueError("Prepared FD001 dataset is empty")

        required_columns = [self.engine_id_column, self.cycle_column]
        if self.target_column is not None:
            required_columns.append(self.target_column)

        missing = [column for column in required_columns if column not in self._frame.columns]
        if missing:
            raise ValueError(f"Prepared FD001 dataset is missing required columns: {missing}")

        if self._frame[[self.engine_id_column, self.cycle_column]].isna().any().any():
            raise ValueError("Prepared FD001 dataset contains null engine_id or cycle values")

        duplicated = self._frame.duplicated([self.engine_id_column, self.cycle_column])
        if duplicated.any():
            count = int(duplicated.sum())
            raise ValueError(f"Prepared FD001 dataset contains {count} duplicate engine-cycle rows")

        engine_ids = list(pd.unique(self._frame[self.engine_id_column]))
        for engine_id in engine_ids:
            rows = self._frame.loc[self._frame[self.engine_id_column] == engine_id]
            self._validate_engine_order(engine_id, rows)

        return engine_ids

    def _validate_engine_order(self, engine_id: Any, rows: pd.DataFrame) -> None:
        cycles = rows[self.cycle_column]
        if not cycles.is_monotonic_increasing:
            raise ValueError(
                f"Cycle values must be monotonically increasing for engine {engine_id!r}"
            )
        if cycles.duplicated().any():
            raise ValueError(f"Duplicate cycle values found for engine {engine_id!r}")
        self.logger.debug(
            "Validated ordering for engine %r with %d rows",
            engine_id,
            len(rows),
        )


def iter_engine_sequences(
    frame: pd.DataFrame,
    *,
    engine_id_column: str = "unit_id",
    cycle_column: str = "cycle",
    target_column: str | None = None,
    copy_rows: bool = True,
    logger: logging.Logger | None = None,
) -> EngineSequenceIterable:
    """Return an iterable over isolated FD001 engine trajectories."""
    return EngineSequenceIterable(
        frame,
        engine_id_column=engine_id_column,
        cycle_column=cycle_column,
        target_column=target_column,
        copy_rows=copy_rows,
        logger=logger,
    )
