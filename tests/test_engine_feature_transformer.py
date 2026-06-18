from __future__ import annotations

import logging
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.engine_feature_transformer import EngineFeatureTransformer
from src.features.engine_sequences import EngineSequence, iter_engine_sequences


class EngineFeatureTransformerTests(unittest.TestCase):
    def test_transform_returns_raw_sensors_and_first_order_deltas(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "cycle": [1, 2],
                "sensor_1": [10.0, 11.0],
                "sensor_2": [50.0, 47.5],
                "rul_capped": [100, 99],
            }
        )
        sequence = next(iter(iter_engine_sequences(frame, target_column="rul_capped")))
        transformer = EngineFeatureTransformer(
            sensor_columns=["sensor_1", "sensor_2"],
            rolling_window_sizes=[5],
        )

        transformed = transformer.transform(sequence)

        expected = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "cycle": [1, 2],
                "sensor_1": [10.0, 11.0],
                "sensor_2": [50.0, 47.5],
                "rul_capped": [100, 99],
                "delta_sensor_1": [0.0, 1.0],
                "sensor_1_rolling_mean_5": [10.0, 10.5],
                "sensor_1_rolling_std_5": [0.0, 0.5],
                "sensor_1_rolling_min_5": [10.0, 10.0],
                "sensor_1_rolling_max_5": [10.0, 11.0],
                "delta_sensor_2": [0.0, -2.5],
                "sensor_2_rolling_mean_5": [50.0, 48.75],
                "sensor_2_rolling_std_5": [0.0, 1.25],
                "sensor_2_rolling_min_5": [50.0, 47.5],
                "sensor_2_rolling_max_5": [50.0, 50.0],
            }
        )
        assert_frame_equal(transformed, expected)
        self.assertEqual(transformer.current_engine_id, 1)
        self.assertEqual(transformer.state["rows_processed"], 2)
        self.assertEqual(transformer.state["last_cycle"], 2)
        self.assertEqual(transformer.state["previous_sensor_values"]["sensor_1"], 11.0)

    def test_transform_resets_state_between_engines(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 2],
                "cycle": [1, 2, 1],
                "sensor_1": [10.0, 11.0, 20.0],
                "rul_capped": [100, 99, 80],
            }
        )
        sequences = list(iter_engine_sequences(frame, target_column="rul_capped"))
        transformer = EngineFeatureTransformer()

        transformer.transform(sequences[0])
        transformer.transform(sequences[1])

        self.assertEqual(transformer.current_engine_id, 2)
        self.assertEqual(transformer.state["rows_processed"], 1)
        self.assertEqual(transformer.state["last_cycle"], 1)
        self.assertEqual(transformer.state["previous_sensor_values"]["sensor_1"], 20.0)

    def test_first_timestep_delta_is_zero_after_each_engine_reset(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "cycle": [1, 2, 1, 2],
                "sensor_1": [10.0, 12.0, 30.0, 35.0],
                "rul_capped": [100, 99, 80, 79],
            }
        )
        sequences = list(iter_engine_sequences(frame, target_column="rul_capped"))
        transformer = EngineFeatureTransformer(sensor_columns=["sensor_1"])

        first_engine = transformer.transform(sequences[0])
        second_engine = transformer.transform(sequences[1])

        self.assertEqual(first_engine["delta_sensor_1"].tolist(), [0.0, 2.0])
        self.assertEqual(second_engine["delta_sensor_1"].tolist(), [0.0, 5.0])
        self.assertEqual(first_engine["sensor_1_rolling_mean_5"].tolist(), [10.0, 11.0])
        self.assertEqual(second_engine["sensor_1_rolling_mean_5"].tolist(), [30.0, 32.5])

    def test_rolling_features_are_causal_and_use_fixed_size_buffer(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 1],
                "cycle": [1, 2, 3],
                "sensor_1": [1.0, 2.0, 4.0],
            }
        )
        sequence = EngineSequence(engine_id=1, rows=frame)
        transformer = EngineFeatureTransformer(
            sensor_columns=["sensor_1"],
            rolling_window_sizes=[2],
        )

        transformed = transformer.transform(sequence)

        self.assertEqual(transformed["sensor_1_rolling_mean_2"].tolist(), [1.0, 1.5, 3.0])
        self.assertEqual(transformed["sensor_1_rolling_std_2"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(transformed["sensor_1_rolling_min_2"].tolist(), [1.0, 1.0, 2.0])
        self.assertEqual(transformed["sensor_1_rolling_max_2"].tolist(), [1.0, 2.0, 4.0])
        rolling_state = transformer.state["rolling"]["sensor_1"][2]
        self.assertEqual(list(rolling_state.values), [2.0, 4.0])
        self.assertLessEqual(len(rolling_state.values), 2)

    def test_update_requires_reset_before_processing(self) -> None:
        transformer = EngineFeatureTransformer()
        row = pd.Series({"unit_id": 1, "cycle": 1, "sensor_1": 10.0})

        with self.assertRaisesRegex(RuntimeError, "call reset"):
            transformer.update(row)

    def test_update_rejects_row_from_different_engine(self) -> None:
        transformer = EngineFeatureTransformer()
        transformer.reset(1)
        row = pd.Series({"unit_id": 2, "cycle": 1, "sensor_1": 20.0})

        with self.assertRaisesRegex(RuntimeError, "state contamination"):
            transformer.update(row)

    def test_transform_rejects_sequence_with_mixed_engine_ids(self) -> None:
        rows = pd.DataFrame(
            {
                "unit_id": [1, 2],
                "cycle": [1, 2],
                "sensor_1": [10.0, 20.0],
            }
        )
        sequence = EngineSequence(engine_id=1, rows=rows)
        transformer = EngineFeatureTransformer()

        with self.assertRaisesRegex(ValueError, "exactly one engine_id"):
            transformer.transform(sequence)

    def test_transform_logs_state_reset_and_processed_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [3, 3],
                "cycle": [1, 2],
                "sensor_1": [30.0, 31.0],
            }
        )
        sequence = EngineSequence(engine_id=3, rows=frame)
        logger = logging.getLogger("test_engine_feature_transformer")
        transformer = EngineFeatureTransformer(logger=logger)

        with self.assertLogs(logger, level="DEBUG") as captured:
            transformer.transform(sequence)

        log_text = "\n".join(captured.output)
        self.assertIn("Reset feature transformer state for engine 3", log_text)
        self.assertIn("Processed 2 rows for engine 3", log_text)

    def test_from_config_uses_configured_sensor_columns(self) -> None:
        config = {
            "columns": {
                "id": "unit_id",
                "cycle": "cycle",
                "sensors": ["sensor_1"],
            }
        }
        frame = pd.DataFrame(
            {
                "unit_id": [4, 4],
                "cycle": [1, 2],
                "sensor_1": [7.0, 10.0],
            }
        )
        sequence = EngineSequence(engine_id=4, rows=frame)
        transformer = EngineFeatureTransformer.from_config(config)

        transformed = transformer.transform(sequence)

        self.assertEqual(transformed["delta_sensor_1"].tolist(), [0.0, 3.0])


if __name__ == "__main__":
    unittest.main()
