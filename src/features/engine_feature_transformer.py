from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.features.engine_sequences import EngineSequence

LOGGER = logging.getLogger(__name__)


class EngineFeatureTransformer:
    """Stateful scaffold for future causal per-engine feature computation.

    The transformer processes one engine sequence at a time in cycle order. The
    current implementation intentionally performs an identity transformation;
    it only establishes reset, update, and transform semantics for later feature
    enrichment.
    """

    def __init__(
        self,
        *,
        engine_id_column: str = "unit_id",
        cycle_column: str = "cycle",
        debug_assertions: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.engine_id_column = engine_id_column
        self.cycle_column = cycle_column
        self.debug_assertions = debug_assertions
        self.logger = logger or LOGGER
        self.current_engine_id: Any | None = None
        self.state: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        debug_assertions: bool = True,
        logger: logging.Logger | None = None,
    ) -> EngineFeatureTransformer:
        """Create a transformer using project FD001 column names from config."""
        columns = config["columns"]
        return cls(
            engine_id_column=columns["id"],
            cycle_column=columns["cycle"],
            debug_assertions=debug_assertions,
            logger=logger,
        )

    def reset(self, engine_id: Any) -> None:
        """Clear all per-engine state before processing a new engine."""
        self.current_engine_id = engine_id
        self.state = {
            "rows_processed": 0,
            "last_cycle": None,
        }
        self.logger.debug("Reset feature transformer state for engine %r", engine_id)

    def update(self, row: pd.Series) -> pd.Series:
        """Process one timestep causally and return the unchanged row.

        Future feature functions should be added inside this method while
        preserving the external reset/update/transform interface.
        """
        if self.current_engine_id is None:
            raise RuntimeError("Transformer state is not initialized; call reset(engine_id) first")

        if self.debug_assertions:
            self._validate_row_consistency(row)

        self.state["rows_processed"] += 1
        self.state["last_cycle"] = row[self.cycle_column]
        return row.copy(deep=True)

    def transform(self, engine_sequence: EngineSequence) -> pd.DataFrame:
        """Transform a single engine sequence in streaming order.

        The method iterates over rows sequentially to simulate time progression.
        It does not inspect future rows to compute the current row, and currently
        returns an identity copy of the input sequence.
        """
        self._validate_engine_sequence(engine_sequence)
        self.reset(engine_sequence.engine_id)

        output = engine_sequence.rows.copy(deep=True)
        for _, row in engine_sequence.rows.iterrows():
            self.update(row)

        self.logger.debug(
            "Processed %d rows for engine %r",
            self.state["rows_processed"],
            self.current_engine_id,
        )
        return output

    def _validate_engine_sequence(self, engine_sequence: EngineSequence) -> None:
        rows = engine_sequence.rows
        if rows.empty:
            raise ValueError(f"Engine sequence {engine_sequence.engine_id!r} is empty")

        missing = [
            column
            for column in [self.engine_id_column, self.cycle_column]
            if column not in rows.columns
        ]
        if missing:
            raise ValueError(f"Engine sequence is missing required columns: {missing}")

        unique_engine_ids = pd.unique(rows[self.engine_id_column])
        if len(unique_engine_ids) != 1 or unique_engine_ids[0] != engine_sequence.engine_id:
            raise ValueError(
                "Engine sequence rows must contain exactly one engine_id matching "
                f"{engine_sequence.engine_id!r}"
            )

        cycles = rows[self.cycle_column]
        if not cycles.is_monotonic_increasing:
            raise ValueError(
                f"Engine sequence {engine_sequence.engine_id!r} is not ordered by cycle"
            )

    def _validate_row_consistency(self, row: pd.Series) -> None:
        row_engine_id = row[self.engine_id_column]
        if row_engine_id != self.current_engine_id:
            raise RuntimeError(
                "Engine state contamination detected: row engine_id "
                f"{row_engine_id!r} does not match active engine "
                f"{self.current_engine_id!r}"
            )

        current_cycle = row[self.cycle_column]
        last_cycle = self.state.get("last_cycle")
        if last_cycle is not None and current_cycle < last_cycle:
            raise RuntimeError(
                "Temporal ordering violation detected while processing engine "
                f"{self.current_engine_id!r}"
            )
