from __future__ import annotations

import unittest

import pandas as pd

from src.features.validation import ValidationSuite


class FeatureValidationSuiteTests(unittest.TestCase):
    def test_validate_all_passes_for_isolated_engineered_splits(self) -> None:
        train = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "cycle": [1, 2],
                "sensor_1": [10.0, 11.0],
                "delta_sensor_1": [0.0, 1.0],
                "sensor_1_rolling_std_5": [0.0, 0.5],
                "sensor_1_slope_5": [0.0, 1.0],
                "rul_capped": [100, 99],
            }
        )
        validation = pd.DataFrame(
            {
                "unit_id": [2, 2],
                "cycle": [1, 2],
                "sensor_1": [20.0, 22.0],
                "delta_sensor_1": [0.0, 2.0],
                "sensor_1_rolling_std_5": [0.0, 1.0],
                "sensor_1_slope_5": [0.0, 2.0],
                "rul_capped": [80, 79],
            }
        )

        report = ValidationSuite().validate_all(
            {"train": train, "validation": validation}
        )

        self.assertTrue(report.passed)
        self.assertTrue(all(check.passed for check in report.checks))
        self.assertTrue(report.as_dict()["passed"])

    def test_temporal_causality_fails_for_non_monotonic_engine_order(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "cycle": [2, 1],
                "sensor_1": [11.0, 10.0],
                "delta_sensor_1": [1.0, 0.0],
            }
        )

        result = ValidationSuite().check_temporal_causality(frame, name="train")

        self.assertFalse(result.passed)
        self.assertIn("cycles are not increasing", "\n".join(result.messages))

    def test_engine_isolation_fails_when_first_row_stateful_features_are_not_reset(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "cycle": [1, 2, 1, 2],
                "sensor_1": [10.0, 11.0, 30.0, 31.0],
                "delta_sensor_1": [0.0, 1.0, 19.0, 1.0],
                "sensor_1_slope_5": [0.0, 1.0, 19.0, 1.0],
            }
        )

        result = ValidationSuite().check_engine_isolation(frame, name="train")

        self.assertFalse(result.passed)
        self.assertIn("first row for engine 2", "\n".join(result.messages))

    def test_split_integrity_fails_for_overlapping_engine_ids(self) -> None:
        train = pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [1],
                "sensor_1": [10.0],
                "delta_sensor_1": [0.0],
            }
        )
        test = pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [1],
                "sensor_1": [20.0],
                "delta_sensor_1": [0.0],
            }
        )

        result = ValidationSuite().check_split_integrity({"train": train, "test": test})

        self.assertFalse(result.passed)
        self.assertIn("overlapping engine IDs", "\n".join(result.messages))

    def test_split_integrity_fails_for_schema_mismatch(self) -> None:
        train = pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [1],
                "sensor_1": [10.0],
                "delta_sensor_1": [0.0],
            }
        )
        validation = pd.DataFrame(
            {
                "unit_id": [2],
                "cycle": [1],
                "sensor_1": [20.0],
            }
        )

        result = ValidationSuite().check_split_integrity(
            {"train": train, "validation": validation}
        )

        self.assertFalse(result.passed)
        self.assertIn("feature schema differs", "\n".join(result.messages))

    def test_temporal_causality_flags_future_looking_feature_names(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1],
                "cycle": [1],
                "sensor_1": [10.0],
                "sensor_1_future_mean": [12.0],
            }
        )

        result = ValidationSuite().check_temporal_causality(frame, name="train")

        self.assertFalse(result.passed)
        self.assertIn("future-looking feature name", "\n".join(result.messages))


if __name__ == "__main__":
    unittest.main()
