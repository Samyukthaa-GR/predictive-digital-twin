from __future__ import annotations

import logging
import unittest

import pandas as pd

from src.features.engine_sequences import EngineSequenceIterable, iter_engine_sequences


class EngineSequenceTests(unittest.TestCase):
    def test_iter_engine_sequences_returns_isolated_ordered_engine_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "cycle": [1, 2, 1, 2],
                "sensor_1": [10.0, 11.0, 20.0, 21.0],
                "rul_capped": [100, 99, 80, 79],
            }
        )

        logger = logging.getLogger("test_engine_sequences")
        with self.assertLogs(logger, level="INFO") as captured:
            sequences = list(
                iter_engine_sequences(
                    frame,
                    target_column="rul_capped",
                    logger=logger,
                )
            )

        self.assertEqual([sequence.engine_id for sequence in sequences], [1, 2])
        self.assertEqual(sequences[0].rows["unit_id"].tolist(), [1, 1])
        self.assertEqual(sequences[0].rows["cycle"].tolist(), [1, 2])
        self.assertEqual(sequences[1].rows["unit_id"].tolist(), [2, 2])
        self.assertEqual(sequences[1].rows["cycle"].tolist(), [1, 2])
        self.assertIn("Prepared 2 isolated engine sequences", "\n".join(captured.output))

    def test_iter_engine_sequences_allows_interleaved_engines_when_each_engine_is_ordered(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 2, 1, 2],
                "cycle": [1, 1, 2, 2],
                "sensor_1": [10.0, 20.0, 11.0, 21.0],
                "rul_capped": [100, 80, 99, 79],
            }
        )

        sequences = list(iter_engine_sequences(frame, target_column="rul_capped"))

        self.assertEqual(
            [sequence.rows["cycle"].tolist() for sequence in sequences],
            [[1, 2], [1, 2]],
        )

    def test_iter_engine_sequences_rejects_non_monotonic_cycles_per_engine(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 2],
                "cycle": [2, 1, 1],
                "sensor_1": [11.0, 10.0, 20.0],
                "rul_capped": [99, 100, 80],
            }
        )

        with self.assertRaisesRegex(ValueError, "monotonically increasing"):
            list(iter_engine_sequences(frame, target_column="rul_capped"))

    def test_iter_engine_sequences_rejects_duplicate_engine_cycle_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "cycle": [1, 1],
                "sensor_1": [10.0, 10.5],
                "rul_capped": [100, 100],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate engine-cycle"):
            list(iter_engine_sequences(frame, target_column="rul_capped"))

    def test_engine_sequence_iterable_from_config_uses_project_column_names(self) -> None:
        config = {
            "columns": {"id": "unit_id", "cycle": "cycle"},
            "labels": {"capped_target": "rul_capped"},
        }
        frame = pd.DataFrame(
            {
                "unit_id": [3, 3],
                "cycle": [1, 2],
                "sensor_1": [30.0, 31.0],
                "rul_capped": [50, 49],
            }
        )

        iterable = EngineSequenceIterable.from_config(frame, config)

        self.assertEqual(len(iterable), 1)
        sequence = next(iter(iterable))
        self.assertEqual(sequence.engine_id, 3)
        self.assertEqual(sequence.rows["cycle"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
