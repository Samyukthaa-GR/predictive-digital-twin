from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.features.engine_sequences import EngineSequence

LOGGER = logging.getLogger(__name__)


@dataclass
class RollingWindowState:
    """Incremental causal rolling and slope state for one sensor/window size."""

    window_size: int
    times: deque[float] = field(default_factory=deque)
    values: deque[float] = field(default_factory=deque)
    min_values: deque[float] = field(default_factory=deque)
    max_values: deque[float] = field(default_factory=deque)
    running_sum_time: float = 0.0
    running_sum_time_squares: float = 0.0
    running_sum_time_value: float = 0.0
    running_sum: float = 0.0
    running_sum_squares: float = 0.0

    def update(self, time: float, value: float) -> dict[str, float]:
        self.times.append(time)
        self.values.append(value)
        self.running_sum_time += time
        self.running_sum_time_squares += time * time
        self.running_sum_time_value += time * value
        self.running_sum += value
        self.running_sum_squares += value * value

        while self.min_values and self.min_values[-1] > value:
            self.min_values.pop()
        self.min_values.append(value)

        while self.max_values and self.max_values[-1] < value:
            self.max_values.pop()
        self.max_values.append(value)

        if len(self.values) > self.window_size:
            expired_time = self.times.popleft()
            expired = self.values.popleft()
            self.running_sum_time -= expired_time
            self.running_sum_time_squares -= expired_time * expired_time
            self.running_sum_time_value -= expired_time * expired
            self.running_sum -= expired
            self.running_sum_squares -= expired * expired
            if self.min_values and self.min_values[0] == expired:
                self.min_values.popleft()
            if self.max_values and self.max_values[0] == expired:
                self.max_values.popleft()

        count = len(self.values)
        mean = self.running_sum / count
        variance = max((self.running_sum_squares / count) - (mean * mean), 0.0)
        denominator = (count * self.running_sum_time_squares) - (
            self.running_sum_time * self.running_sum_time
        )
        # A single available point has no identifiable trend; use a neutral
        # causal fallback until at least two observations are available.
        slope = 0.0
        if count >= 2 and denominator != 0:
            slope = (
                (count * self.running_sum_time_value) - (self.running_sum_time * self.running_sum)
            ) / denominator
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min_values[0],
            "max": self.max_values[0],
            "slope": slope,
        }


class EngineFeatureTransformer:
    """Stateful transformer for causal per-engine feature computation.

    The transformer processes one engine sequence at a time in cycle order. The
    current implementation passes raw sensor values through unchanged and adds
    first-order sensor deltas, causal rolling statistics, and causal rolling
    trend slopes. Normalization and modeling logic are intentionally out of
    scope.
    """

    def __init__(
        self,
        *,
        engine_id_column: str = "unit_id",
        cycle_column: str = "cycle",
        sensor_columns: list[str] | tuple[str, ...] | None = None,
        rolling_window_sizes: list[int] | tuple[int, ...] = (5, 10, 20),
        debug_assertions: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.engine_id_column = engine_id_column
        self.cycle_column = cycle_column
        self.sensor_columns = tuple(sensor_columns) if sensor_columns is not None else None
        self.rolling_window_sizes = tuple(rolling_window_sizes)
        if not self.rolling_window_sizes or any(size <= 0 for size in self.rolling_window_sizes):
            raise ValueError("rolling_window_sizes must contain positive integer window sizes")
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
            sensor_columns=columns["sensors"],
            rolling_window_sizes=(5, 10, 20),
            debug_assertions=debug_assertions,
            logger=logger,
        )

    def reset(self, engine_id: Any) -> None:
        """Clear all per-engine state before processing a new engine."""
        self.current_engine_id = engine_id
        self.state = {
            "rows_processed": 0,
            "last_cycle": None,
            "previous_sensor_values": None,
            "rolling": {},
        }
        self.logger.debug("Reset feature transformer state for engine %r", engine_id)

    def update(self, row: pd.Series) -> pd.Series:
        """Process one timestep causally and return raw plus delta features.

        Future feature functions should be added inside this method while
        preserving the external reset/update/transform interface.
        """
        if self.current_engine_id is None:
            raise RuntimeError("Transformer state is not initialized; call reset(engine_id) first")

        if self.debug_assertions:
            self._validate_row_consistency(row)

        sensor_columns = self._resolve_sensor_columns(row.index)
        current_sensor_values = row.loc[list(sensor_columns)].copy(deep=True)
        previous_sensor_values = self.state["previous_sensor_values"]

        transformed = row.copy(deep=True)
        for sensor_column in sensor_columns:
            delta_column = self._delta_column_name(sensor_column)
            if previous_sensor_values is None:
                transformed[delta_column] = 0.0
            else:
                transformed[delta_column] = (
                    current_sensor_values[sensor_column] - previous_sensor_values[sensor_column]
                )
            self._add_rolling_features(
                transformed,
                sensor_column,
                float(row[self.cycle_column]),
                float(current_sensor_values[sensor_column]),
            )

        self.state["rows_processed"] += 1
        self.state["last_cycle"] = row[self.cycle_column]
        self.state["previous_sensor_values"] = current_sensor_values
        return transformed

    def transform(self, engine_sequence: EngineSequence) -> pd.DataFrame:
        """Transform a single engine sequence in streaming order.

        The method iterates over rows sequentially to simulate time progression.
        It does not inspect future rows to compute the current row.
        """
        self._validate_engine_sequence(engine_sequence)
        self.reset(engine_sequence.engine_id)

        output_rows: list[pd.Series] = []
        for _, row in engine_sequence.rows.iterrows():
            output_rows.append(self.update(row))

        self.logger.debug(
            "Processed %d rows for engine %r",
            self.state["rows_processed"],
            self.current_engine_id,
        )
        output = pd.DataFrame(output_rows, index=engine_sequence.rows.index)
        for column, dtype in engine_sequence.rows.dtypes.items():
            output[column] = output[column].astype(dtype)
        return output

    def _validate_engine_sequence(self, engine_sequence: EngineSequence) -> None:
        rows = engine_sequence.rows
        if rows.empty:
            raise ValueError(f"Engine sequence {engine_sequence.engine_id!r} is empty")

        sensor_columns = self._resolve_sensor_columns(rows.columns)
        missing = [
            column
            for column in [self.engine_id_column, self.cycle_column, *sensor_columns]
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
        if cycles.duplicated().any():
            raise ValueError(
                f"Engine sequence {engine_sequence.engine_id!r} contains duplicate cycles"
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
        if last_cycle is not None and current_cycle <= last_cycle:
            raise RuntimeError(
                "Temporal ordering violation detected while processing engine "
                f"{self.current_engine_id!r}"
            )
        for sensor_state in self.state.get("rolling", {}).values():
            for rolling_state in sensor_state.values():
                if len(rolling_state.values) > rolling_state.window_size:
                    raise RuntimeError("Rolling buffer exceeded configured window size")
                if len(rolling_state.times) != len(rolling_state.values):
                    raise RuntimeError("Rolling time/value buffer length mismatch")

    def _resolve_sensor_columns(self, available_columns) -> tuple[str, ...]:
        if self.sensor_columns is not None:
            return self.sensor_columns

        inferred = tuple(column for column in available_columns if str(column).startswith("sensor_"))
        if not inferred:
            raise ValueError(
                "No sensor columns were configured or inferred; provide sensor_columns explicitly"
            )
        self.sensor_columns = inferred
        return inferred

    @staticmethod
    def _delta_column_name(sensor_column: str) -> str:
        return f"delta_{sensor_column}"

    def _add_rolling_features(
        self,
        transformed: pd.Series,
        sensor_column: str,
        cycle: float,
        sensor_value: float,
    ) -> None:
        rolling_by_sensor = self.state["rolling"].setdefault(sensor_column, {})
        for window_size in self.rolling_window_sizes:
            rolling_state = rolling_by_sensor.setdefault(
                window_size,
                RollingWindowState(window_size=window_size),
            )
            stats = rolling_state.update(cycle, sensor_value)
            if self.debug_assertions and len(rolling_state.values) > window_size:
                raise RuntimeError("Rolling buffer exceeded configured window size")
            if self.debug_assertions and len(rolling_state.times) != len(rolling_state.values):
                raise RuntimeError("Rolling time/value buffer length mismatch")

            transformed[self._rolling_column_name(sensor_column, window_size, "mean")] = stats[
                "mean"
            ]
            transformed[self._rolling_column_name(sensor_column, window_size, "std")] = stats[
                "std"
            ]
            transformed[self._rolling_column_name(sensor_column, window_size, "min")] = stats[
                "min"
            ]
            transformed[self._rolling_column_name(sensor_column, window_size, "max")] = stats[
                "max"
            ]
            transformed[self._slope_column_name(sensor_column, window_size)] = stats["slope"]

    @staticmethod
    def _rolling_column_name(sensor_column: str, window_size: int, statistic: str) -> str:
        return f"{sensor_column}_rolling_{statistic}_{window_size}"

    @staticmethod
    def _slope_column_name(sensor_column: str, window_size: int) -> str:
        return f"{sensor_column}_slope_{window_size}"
