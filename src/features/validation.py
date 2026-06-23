from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class ValidationCheckResult:
    """Result for one validation check."""

    name: str
    passed: bool
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """Structured leakage and correctness validation report."""

    checks: tuple[ValidationCheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "messages": list(check.messages),
                }
                for check in self.checks
            ],
        }


@dataclass(frozen=True)
class ValidationSuite:
    """Standalone validation stage for engineered FD001 feature datasets.

    The suite inspects already engineered datasets. It does not compute,
    transform, normalize, model, or reconstruct features.
    """

    engine_id_column: str = "unit_id"
    cycle_column: str = "cycle"
    neutral_prefixes: tuple[str, ...] = ("delta_",)
    neutral_substrings: tuple[str, ...] = ("_slope_", "_rolling_std_")
    suspicious_future_terms: tuple[str, ...] = (
        "future",
        "lead",
        "next_",
        "t_plus",
        "t+",
        "centered",
        "centre",
        "lookahead",
    )

    @classmethod
    def from_config(cls, config: dict) -> ValidationSuite:
        columns = config["columns"]
        return cls(
            engine_id_column=columns["id"],
            cycle_column=columns["cycle"],
        )

    def validate_engineered_dataset(self, frame: pd.DataFrame, *, name: str = "dataset") -> ValidationReport:
        """Validate one engineered split for temporal and engine-isolation risks."""
        checks = (
            self.check_temporal_causality(frame, name=name),
            self.check_engine_isolation(frame, name=name),
        )
        return ValidationReport(checks=checks)

    def validate_splits(self, splits: Mapping[str, pd.DataFrame]) -> ValidationReport:
        """Validate train/validation/test split isolation."""
        return ValidationReport(checks=(self.check_split_integrity(splits),))

    def validate_all(self, splits: Mapping[str, pd.DataFrame]) -> ValidationReport:
        """Validate each engineered split plus cross-split integrity."""
        checks: list[ValidationCheckResult] = []
        for split_name, frame in splits.items():
            checks.append(self.check_temporal_causality(frame, name=split_name))
            checks.append(self.check_engine_isolation(frame, name=split_name))
        checks.append(self.check_split_integrity(splits))
        return ValidationReport(checks=tuple(checks))

    def check_temporal_causality(
        self,
        frame: pd.DataFrame,
        *,
        name: str = "dataset",
    ) -> ValidationCheckResult:
        """Inspect temporal ordering and future-looking feature red flags."""
        messages = self._required_column_errors(frame, name)
        if messages:
            return ValidationCheckResult("temporal_causality", False, tuple(messages))

        messages.extend(self._engine_cycle_key_errors(frame, name))
        messages.extend(self._monotonic_cycle_errors(frame, name))
        messages.extend(self._future_feature_name_errors(frame, name))
        messages.extend(self._first_row_neutral_state_errors(frame, name))

        return ValidationCheckResult("temporal_causality", not messages, tuple(messages))

    def check_engine_isolation(
        self,
        frame: pd.DataFrame,
        *,
        name: str = "dataset",
    ) -> ValidationCheckResult:
        """Inspect engine boundary correctness and state-reset assumptions."""
        messages = self._required_column_errors(frame, name)
        if messages:
            return ValidationCheckResult("engine_isolation", False, tuple(messages))

        messages.extend(self._engine_cycle_key_errors(frame, name))
        messages.extend(self._monotonic_cycle_errors(frame, name))
        messages.extend(self._first_row_neutral_state_errors(frame, name))

        engine_count = frame[self.engine_id_column].nunique(dropna=True)
        if engine_count == 0:
            messages.append(f"{name}: no engine groups found")

        return ValidationCheckResult("engine_isolation", not messages, tuple(messages))

    def check_split_integrity(
        self,
        splits: Mapping[str, pd.DataFrame],
    ) -> ValidationCheckResult:
        """Ensure no engine identifiers overlap across engineered splits."""
        messages: list[str] = []
        if not splits:
            return ValidationCheckResult(
                "split_integrity",
                False,
                ("No splits were provided for validation",),
            )

        engine_ids_by_split: dict[str, set] = {}
        feature_columns_by_split: dict[str, set[str]] = {}
        for split_name, frame in splits.items():
            required_errors = self._required_column_errors(frame, split_name)
            messages.extend(required_errors)
            if required_errors:
                continue

            engine_ids_by_split[split_name] = set(frame[self.engine_id_column].dropna().unique())
            feature_columns_by_split[split_name] = set(frame.columns) - {
                self.engine_id_column,
                self.cycle_column,
            }

        split_names = list(engine_ids_by_split)
        for left_index, left_name in enumerate(split_names):
            for right_name in split_names[left_index + 1 :]:
                overlap = engine_ids_by_split[left_name].intersection(engine_ids_by_split[right_name])
                if overlap:
                    messages.append(
                        f"{left_name}/{right_name}: overlapping engine IDs detected: "
                        f"{sorted(overlap)}"
                    )

        if feature_columns_by_split:
            reference_name = next(iter(feature_columns_by_split))
            reference_columns = feature_columns_by_split[reference_name]
            for split_name, columns in feature_columns_by_split.items():
                missing = sorted(reference_columns - columns)
                extra = sorted(columns - reference_columns)
                if missing or extra:
                    messages.append(
                        f"{split_name}: feature schema differs from {reference_name}; "
                        f"missing={missing}, extra={extra}"
                    )

        return ValidationCheckResult("split_integrity", not messages, tuple(messages))

    def _required_column_errors(self, frame: pd.DataFrame, name: str) -> list[str]:
        missing = [
            column
            for column in (self.engine_id_column, self.cycle_column)
            if column not in frame.columns
        ]
        if missing:
            return [f"{name}: missing required columns: {missing}"]
        if frame.empty:
            return [f"{name}: engineered dataset is empty"]
        if frame[[self.engine_id_column, self.cycle_column]].isna().any().any():
            return [f"{name}: null engine_id or cycle values detected"]
        return []

    def _engine_cycle_key_errors(self, frame: pd.DataFrame, name: str) -> list[str]:
        duplicated = frame.duplicated([self.engine_id_column, self.cycle_column])
        if duplicated.any():
            return [
                f"{name}: {int(duplicated.sum())} duplicate engine-cycle rows detected"
            ]
        return []

    def _monotonic_cycle_errors(self, frame: pd.DataFrame, name: str) -> list[str]:
        messages: list[str] = []
        for engine_id, group in frame.groupby(self.engine_id_column, sort=False):
            cycles = group[self.cycle_column]
            if not cycles.is_monotonic_increasing:
                messages.append(f"{name}: cycles are not increasing for engine {engine_id!r}")
            if cycles.duplicated().any():
                messages.append(f"{name}: duplicate cycles detected for engine {engine_id!r}")
        return messages

    def _future_feature_name_errors(self, frame: pd.DataFrame, name: str) -> list[str]:
        messages: list[str] = []
        for column in frame.columns:
            normalized = str(column).lower()
            if any(term in normalized for term in self.suspicious_future_terms):
                messages.append(
                    f"{name}: suspicious future-looking feature name detected: {column!r}"
                )
        return messages

    def _first_row_neutral_state_errors(self, frame: pd.DataFrame, name: str) -> list[str]:
        stateful_columns = self._neutral_state_columns(frame)
        if not stateful_columns:
            return []

        messages: list[str] = []
        for engine_id, group in frame.groupby(self.engine_id_column, sort=False):
            first_row = group.iloc[0]
            non_neutral = [
                column
                for column in stateful_columns
                if pd.notna(first_row[column]) and first_row[column] != 0
            ]
            if non_neutral:
                messages.append(
                    f"{name}: first row for engine {engine_id!r} has non-neutral "
                    f"stateful features: {non_neutral}"
                )
        return messages

    def _neutral_state_columns(self, frame: pd.DataFrame) -> list[str]:
        columns: list[str] = []
        for column in frame.columns:
            column_text = str(column)
            if column_text.startswith(self.neutral_prefixes) or any(
                substring in column_text for substring in self.neutral_substrings
            ):
                columns.append(column)
        return columns
