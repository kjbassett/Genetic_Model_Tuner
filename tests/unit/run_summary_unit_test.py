"""A run publishes a summary, and the winner stays in its own folder.

The version folder used to hold a copy of the winning organism. It now holds a
summary that points at where the winner actually lives, so there is exactly one
copy of any organism's state and every organism of every generation is recorded
even after its folder is pruned.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ezmt.hyperparameters import DiscreteNonOrdinal
from ezmt.model_tuner import RUN_SUMMARY_FILE, ModelTuner
from ezmt.organism import Organism
from tests.unit.sequential_unit_test import build_model_space


class RunSummaryTestCase(unittest.IsolatedAsyncioTestCase):
    """Reuses the sequential suite's model space, which is already shaped for
    a population that shares a prefix and forks on one hyperparameter."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def new_tuner(self, n_organisms=4, **kwargs):
        options = {
            "sequential": True,
            "goal": "max",
            "directory": self.directory,
            "temp_directory": f"{self.directory}/.tmp",
        }
        options.update(kwargs)
        return ModelTuner(
            build_model_space(),
            {"mode": DiscreteNonOrdinal(["a", "b"])},
            pop_size=n_organisms,
            **options,
        )

    async def run_tuner(self, modes=("a", "a", "b", "b"), **kwargs):
        tuner = self.new_tuner(n_organisms=len(modes), **kwargs)
        with patch(
            "ezmt.model_tuner.choose_hyperparams",
            side_effect=[{"mode": mode} for mode in modes] * 4,
        ):
            best, summary = await tuner.run("test")
        return tuner, best, summary


class TestRunSummary(RunSummaryTestCase):

    async def test_the_summary_is_written_at_the_version_level(self):
        tuner, _best, _summary = await self.run_tuner()
        self.assertTrue(
            os.path.isfile(os.path.join(tuner.run_folder(), RUN_SUMMARY_FILE)))

    async def test_it_records_every_organism_of_every_generation(self):
        # Arrange - this is the record that outlives the pruned folders.
        _tuner, _best, summary = await self.run_tuner(generations=2)
        # Act
        detail = summary["generations_detail"]
        # Assert
        self.assertEqual(len(detail), 2)
        for generation in detail:
            with self.subTest(generation=generation["generation"]):
                self.assertEqual(len(generation["organisms"]), 4)

    async def test_every_organism_entry_carries_what_the_database_needs(self):
        _tuner, _best, summary = await self.run_tuner()
        entry = summary["generations_detail"][0]["organisms"][0]
        for field in ("generation", "organism_index", "score", "fitness",
                      "dna", "parameters", "folder"):
            with self.subTest(field=field):
                self.assertIn(field, entry)

    async def test_the_best_is_a_pointer_not_a_copy(self):
        # Arrange - the winner stays where it ran; the summary says where.
        tuner, _best, summary = await self.run_tuner()
        # Act
        pointed_at = os.path.join(tuner.run_folder(), summary["best"]["folder"])
        # Assert
        self.assertTrue(os.path.isdir(pointed_at))
        self.assertFalse(
            os.path.isfile(os.path.join(tuner.run_folder(), "knowledge.json")),
            "the version folder should hold no organism state of its own",
        )

    async def test_the_best_is_the_highest_score_across_all_generations(self):
        # Arrange - not just the final one. Without elitism a generation can end
        # worse than one before it, and the run should not return something it
        # already beat.
        _tuner, _best, summary = await self.run_tuner(generations=2)
        every = [o["score"] for g in summary["generations_detail"]
                 for o in g["organisms"]]
        self.assertEqual(summary["best"]["score"], max(every))

    async def test_the_returned_organism_matches_the_summary(self):
        _tuner, best, summary = await self.run_tuner()
        self.assertEqual(best.score, summary["best"]["score"])

    async def test_the_returned_organism_has_its_knowledge(self):
        # Arrange - callers read predictions and metrics off it.
        _tuner, best, _summary = await self.run_tuner()
        self.assertIsInstance(best.knowledge.get("frame"), pd.DataFrame)

    async def test_the_summary_survives_a_round_trip_through_json(self):
        # Arrange - it is written with default=str, so anything unserialisable
        # would land as a string rather than raising here.
        tuner, _best, summary = await self.run_tuner()
        with open(os.path.join(tuner.run_folder(), RUN_SUMMARY_FILE)) as f:
            written = json.load(f)
        self.assertEqual(written["best"]["folder"], summary["best"]["folder"])


class TestLoadingFromTheSummary(RunSummaryTestCase):

    async def test_latest_skips_generation_folders(self):
        # Arrange - os.listdir()[-1] would sort a generation folder named "2"
        # last and resolve a run name to it.
        _tuner, best, _summary = await self.run_tuner(
            generations=2, save_organisms="all")
        # Act
        loaded = Organism.load("test", "latest", directory=self.directory)
        # Assert
        self.assertEqual(loaded.dna, best.dna)

    async def test_a_specific_organism_can_be_loaded_by_position(self):
        # Arrange - the point of recording generation and index.
        _tuner, _best, summary = await self.run_tuner(save_organisms="all")
        entry = summary["generations_detail"][0]["organisms"][0]
        # Act
        loaded = Organism.load(
            "test", summary["version"], directory=self.directory,
            generation=entry["generation"], organism_index=entry["organism_index"],
        )
        # Assert
        self.assertIsInstance(loaded.knowledge.get("frame"), pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
