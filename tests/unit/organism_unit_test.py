import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

from ezmt.organism import Organism


def save_marker(folder, key, value):
    """A save_load_func that writes its own side file, like a torch state dict."""
    path = os.path.join(folder, f"{key}.marker")
    with open(path, "w") as f:
        f.write(str(value))
    return f"{key}.marker"


class TestSaveReturnsThePersistedState(unittest.TestCase):
    """save/save_state hand back the state as files and scalars.

    Sequential runs keep this dict and drop their reference to the live objects,
    which is the only thing that releases a trained model and its datasets from
    memory. So the return value has to be complete and free of live objects, not
    merely convenient.
    """

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def _organism(self, knowledge, save_load_funcs=None):
        return Organism(
            "test",
            [],
            {},
            knowledge=knowledge,
            save_load_funcs=save_load_funcs,
            directory=self.directory,
        )

    def test_scalars_come_back_unchanged(self):
        # Arrange - score_fitness reads data["score"] off this dict, so a
        # JSON-native value has to survive as itself.
        organism = self._organism({"score": 0.25, "epoch": 3, "name": "abc"})
        # Act
        saved = organism.save()
        # Assert
        self.assertEqual(saved["score"], 0.25)
        self.assertEqual(saved["epoch"], 3)
        self.assertEqual(saved["name"], "abc")

    def test_dataframes_come_back_as_file_names(self):
        # Arrange - ThePickler substitutes frames for files inside json.dump and
        # never reports that back, which is why the return value is re-read from
        # the written file rather than assembled in save_state.
        organism = self._organism({"frame": pd.DataFrame({"a": [1, 2, 3]})})
        # Act
        saved = organism.save()
        # Assert
        self.assertIsInstance(saved["frame"], str)
        self.assertTrue(
            os.path.exists(os.path.join(organism.folder, saved["frame"])),
            "the returned name must point at a file that actually exists",
        )

    def test_no_live_object_survives_in_the_returned_state(self):
        # Arrange - anything non-JSON left in here would keep its memory alive.
        organism = self._organism(
            {"frame": pd.DataFrame({"a": [1]}), "obj": object(), "score": 1.0}
        )
        # Act
        saved = organism.save()
        # Assert - round-tripping through json proves nothing live remains
        self.assertEqual(json.loads(json.dumps(saved)), saved)

    def test_custom_save_functions_are_reflected_in_the_return_value(self):
        # Arrange
        organism = self._organism(
            {"weights": [1, 2, 3]}, save_load_funcs={"weights": {"save": save_marker}}
        )
        # Act
        saved = organism.save()
        # Assert
        self.assertEqual(saved["weights"], "weights.marker")

    def test_the_returned_state_matches_the_file_on_disk(self):
        # Arrange - Organism.load reads knowledge.json, so a caller holding the
        # return value and a caller reloading the organism must agree.
        organism = self._organism({"frame": pd.DataFrame({"a": [1]}), "score": 2.0})
        # Act
        saved = organism.save()
        with open(os.path.join(organism.folder, "knowledge.json")) as f:
            on_disk = json.load(f)
        # Assert
        self.assertEqual(saved, on_disk)

    def test_an_organism_without_knowledge_returns_an_empty_dict(self):
        # Arrange - save() skips knowledge.json entirely when there is nothing to
        # write, so there is no file to read back.
        organism = self._organism({})
        # Act
        saved = organism.save()
        # Assert
        self.assertEqual(saved, {})

    def test_saving_does_not_mutate_the_live_knowledge(self):
        # Arrange - the organism may still be used after saving; replacing its
        # frames with file names in place would break that.
        frame = pd.DataFrame({"a": [1]})
        organism = self._organism({"frame": frame})
        # Act
        organism.save()
        # Assert
        self.assertIs(organism.knowledge["frame"], frame)


if __name__ == "__main__":
    unittest.main()
