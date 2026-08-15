import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ezmt.the_pickler import ThePickler, load_frame


class TestFrameRoundTrip(unittest.TestCase):
    """Frames must come back as the type and content they went in as."""

    def _save(self, folder, obj):
        encoded = json.dumps({"frame": obj}, cls=ThePickler, folder=folder)
        return json.loads(encoded)["frame"]

    def test_dataframe_round_trips_via_parquet(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5]})
        with tempfile.TemporaryDirectory() as folder:
            # Act
            name = self._save(folder, df)
            back = load_frame(os.path.join(folder, name))
        # Assert
        self.assertTrue(name.endswith(".parquet"))
        pd.testing.assert_frame_equal(back, df)

    def test_series_round_trips_as_a_series(self):
        # Arrange - parquet has no Series type, so it is wrapped and unwrapped.
        series = pd.Series([1.0, 2.0, 3.0], name="vals")
        with tempfile.TemporaryDirectory() as folder:
            # Act
            name = self._save(folder, series)
            back = load_frame(os.path.join(folder, name))
        # Assert
        self.assertIsInstance(back, pd.Series)
        np.testing.assert_array_equal(back.values, series.values)

    def test_index_is_preserved(self):
        # Arrange - the CSV path used index_col=0; parquet must not lose it.
        df = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
        with tempfile.TemporaryDirectory() as folder:
            # Act
            back = load_frame(os.path.join(folder, self._save(folder, df)))
        # Assert
        self.assertEqual(back.index.tolist(), [10, 20])

    def test_dtypes_survive_unlike_csv(self):
        # Arrange - CSV stringifies everything; parquet keeps the schema.
        df = pd.DataFrame({"i": [1, 2], "f": [1.5, 2.5], "b": [True, False]})
        with tempfile.TemporaryDirectory() as folder:
            # Act
            back = load_frame(os.path.join(folder, self._save(folder, df)))
        # Assert
        self.assertEqual(back["i"].dtype, df["i"].dtype)
        self.assertEqual(back["b"].dtype, df["b"].dtype)

    def test_integer_column_names_round_trip(self):
        # Arrange - pyarrow stores column-name types in metadata, so these
        # survive rather than silently becoming the strings "0" and "1".
        df = pd.DataFrame([[1, 2], [3, 4]], columns=[0, 1])
        with tempfile.TemporaryDirectory() as folder:
            # Act
            back = load_frame(os.path.join(folder, self._save(folder, df)))
        # Assert
        self.assertEqual(list(back.columns), [0, 1])

    def test_mixed_dtype_column_falls_back_to_csv(self):
        # Arrange - parquet raises ArrowInvalid on a column mixing int, str and
        # float, which CSV tolerates. The save must degrade rather than lose the
        # organism's state.
        df = pd.DataFrame({"mixed": [1, "two", 3.0]})
        with tempfile.TemporaryDirectory() as folder:
            # Act
            name = self._save(folder, df)
            # Assert - inside the block; the directory is gone once it exits.
            self.assertTrue(name.endswith(".csv"))
            self.assertTrue(os.path.exists(os.path.join(folder, name)))
            back = load_frame(os.path.join(folder, name))
        self.assertEqual(len(back), 3)


class TestLegacyCsvStillLoads(unittest.TestCase):
    def test_csv_written_before_the_switch_is_readable(self):
        # Arrange - organisms saved prior to parquet must keep working.
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "legacy.csv")
            pd.DataFrame({"a": [1, 2]}).to_csv(path)
            # Act
            back = load_frame(path)
        # Assert
        self.assertEqual(back["a"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
