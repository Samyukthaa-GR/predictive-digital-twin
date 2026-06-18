from __future__ import annotations

import logging
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.engine_feature_transformer import EngineFeatureTransformer
from src.features.engine_sequences import EngineSequence, iter_engine_sequences


class EngineFeatureTransformerTests(unittest.TestCase):
    def test_transform_returns_identity_frame_for_single_engine_sequence(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "cycle": [1, 2],
                "sensor_1": [10.0, 11.0],
                "rul_capped": [100, 99],
            }
        )
        sequence = next(iter(iter_engine_sequences(frame, target_column="rul_capped")))
        transformer = EngineFeatureTransformer()

        transformed = transformer.transform(sequence)

        assert_frame_equal(transformed, sequence.rows)
        self.assertEqual(transformer.current_engine_id, 1)
        self.assertEqual(transformer.state["rows_processed"], 2)
        self.assertEqual(transformer.state["last_cycle"], 2)

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


if __name__ == "__main__":
    unittest.main()
