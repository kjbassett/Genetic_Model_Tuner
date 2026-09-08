"""A run records itself, so a pruned population still has a record.

Losing organisms are deleted from disk as the run goes, so these rows are the
only lasting evidence they existed.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from ezmt.storage import MODEL_TABLE, RUN_TABLE, TrainingStore


def a_summary(scores=((0.1, -0.2), (0.3, 0.05)), best=(2, 0), name="run",
              version="2026-01-01_00-00-00"):
    """A run summary shaped like ModelTuner.run's."""
    detail = [
        {
            "generation": n + 1,
            "organisms": [
                {"generation": n + 1, "organism_index": i, "score": score,
                 "fitness": 0.5, "dna": f"gene({i})", "parameters": {"mode": i},
                 "folder": f"{n + 1}/{i}"}
                for i, score in enumerate(generation)
            ],
        }
        for n, generation in enumerate(scores)
    ]
    generation, index = best
    winner = detail[generation - 1]["organisms"][index]
    return {
        "run_name": name, "version": version, "started_at": 1000,
        "finished_at": 2000, "pop_size": len(scores[0]), "generations": len(scores),
        "goal": "max",
        "best": {"generation": generation, "organism_index": index,
                 "score": winner["score"], "folder": winner["folder"]},
        "generations_detail": detail,
    }


class StorageTestCase(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.path = os.path.join(self.directory, "training.sqlite")
        self.store = TrainingStore(self.path)

    def rows(self, sql, params=()):
        connection = sqlite3.connect(self.path)
        try:
            return connection.execute(sql, params).fetchall()
        finally:
            connection.close()


class TestSaveRun(StorageTestCase):

    async def test_it_creates_its_tables_on_first_use(self):
        # Arrange - a host should not have to run a migration first.
        await self.store.save_run(a_summary())
        # Assert
        names = {r[0] for r in self.rows(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn(RUN_TABLE, names)
        self.assertIn(MODEL_TABLE, names)

    async def test_every_organism_of_every_generation_is_recorded(self):
        # Arrange - two generations of two.
        await self.store.save_run(a_summary())
        # Assert
        self.assertEqual(self.rows(f"SELECT COUNT(*) FROM {MODEL_TABLE}")[0][0], 4)

    async def test_exactly_one_organism_is_the_winner(self):
        await self.store.save_run(a_summary())
        self.assertEqual(
            self.rows(f"SELECT SUM(is_winner) FROM {MODEL_TABLE}")[0][0], 1)

    async def test_the_winner_is_the_one_the_summary_names(self):
        summary = a_summary(best=(1, 0))
        await self.store.save_run(summary)
        row = self.rows(
            f"SELECT generation, organism_index FROM {MODEL_TABLE} WHERE is_winner=1")
        self.assertEqual(row[0], (1, 0))

    async def test_every_organism_points_at_the_run(self):
        result = await self.store.save_run(a_summary())
        linked = self.rows(
            f"SELECT COUNT(*) FROM {MODEL_TABLE} WHERE training_run_id = ?",
            (result["training_run_id"],))
        self.assertEqual(linked[0][0], 4)

    async def test_notes_are_stored_on_the_run_not_each_organism(self):
        # Arrange - the note describes the run, and a population would repeat it
        # once per organism.
        await self.store.save_run(a_summary(), notes="testing offset 7")
        # Assert
        self.assertEqual(
            self.rows(f"SELECT notes FROM {RUN_TABLE}")[0][0], "testing offset 7")

    async def test_parameters_round_trip_as_json(self):
        await self.store.save_run(a_summary())
        stored = self.rows(
            f"SELECT parameters FROM {MODEL_TABLE} ORDER BY organism_index")[0][0]
        self.assertEqual(json.loads(stored), {"mode": 0})

    async def test_it_returns_the_winners_row_id(self):
        result = await self.store.save_run(a_summary())
        self.assertEqual(
            result["winner_model_id"],
            self.rows(f"SELECT id FROM {MODEL_TABLE} WHERE is_winner=1")[0][0])

    async def test_a_single_organism_run_is_recorded(self):
        # Arrange - edge case: one generation, one organism.
        await self.store.save_run(a_summary(scores=((0.5,),), best=(1, 0)))
        self.assertEqual(self.rows(f"SELECT COUNT(*) FROM {MODEL_TABLE}")[0][0], 1)

    async def test_an_organism_that_never_scored_is_still_recorded(self):
        # Arrange - a crashed organism is a result about the search space.
        await self.store.save_run(a_summary(scores=((0.1, None),), best=(1, 0)))
        self.assertEqual(self.rows(f"SELECT COUNT(*) FROM {MODEL_TABLE}")[0][0], 2)

    async def test_two_runs_of_one_name_do_not_collide(self):
        # Arrange - the same organism name is reused every run.
        await self.store.save_run(a_summary(version="v1"))
        await self.store.save_run(a_summary(version="v2"))
        self.assertEqual(self.rows(f"SELECT COUNT(*) FROM {RUN_TABLE}")[0][0], 2)

    async def test_recording_the_same_run_twice_is_rejected(self):
        # Arrange - invalid input: the run is unique on name and version, and a
        # duplicate would double the population.
        await self.store.save_run(a_summary())
        with self.assertRaises(sqlite3.IntegrityError):
            await self.store.save_run(a_summary())


class TestReadingBack(StorageTestCase):

    async def test_the_winner_resolves_by_exact_version(self):
        await self.store.save_run(a_summary(version="v1"))
        self.assertIsNotNone(await self.store.get_winner_id("run", "v1"))

    async def test_latest_resolves_the_most_recent_run(self):
        # Arrange - only winners answer to a name; a loser would be returned by
        # any query that just takes the newest row.
        await self.store.save_run(a_summary(version="v1"))
        second = await self.store.save_run(a_summary(version="v2"))
        self.assertEqual(await self.store.get_winner_id("run", "latest"),
                         second["winner_model_id"])

    async def test_an_unknown_run_has_no_winner(self):
        await self.store.save_run(a_summary())
        self.assertIsNone(await self.store.get_winner_id("nope", "v1"))

    async def test_the_population_comes_back_in_order(self):
        result = await self.store.save_run(a_summary())
        population = await self.store.get_population(result["training_run_id"])
        self.assertEqual([(o["generation"], o["organism_index"]) for o in population],
                         [(1, 0), (1, 1), (2, 0), (2, 1)])

    async def test_the_population_carries_the_scores(self):
        # Arrange - the whole reason these rows exist: the folders are pruned.
        result = await self.store.save_run(a_summary())
        population = await self.store.get_population(result["training_run_id"])
        self.assertEqual([o["score"] for o in population], [0.1, -0.2, 0.3, 0.05])

    async def test_an_unknown_run_has_no_population(self):
        await self.store.save_run(a_summary())
        self.assertEqual(await self.store.get_population(999), [])


class TestHostMetrics(StorageTestCase):

    async def test_metrics_attach_to_the_winner(self):
        # Arrange - only the winner is loaded back with its knowledge, so its
        # metrics arrive after the population is written.
        result = await self.store.save_run(a_summary())
        # Act
        await self.store.update_model_metrics(
            result["winner_model_id"], {"outliers": 3}, {"precision": 0.42})
        # Assert
        row = self.rows(
            f"SELECT data_quality_metrics, model_performance FROM {MODEL_TABLE}"
            f" WHERE id = ?", (result["winner_model_id"],))[0]
        self.assertEqual(json.loads(row[0]), {"outliers": 3})
        self.assertEqual(json.loads(row[1]), {"precision": 0.42})

    async def test_no_metrics_stores_null_rather_than_empty_json(self):
        result = await self.store.save_run(a_summary())
        await self.store.update_model_metrics(result["winner_model_id"])
        row = self.rows(f"SELECT data_quality_metrics FROM {MODEL_TABLE}"
                        f" WHERE id = ?", (result["winner_model_id"],))[0]
        self.assertIsNone(row[0])

    async def test_updating_an_unknown_row_changes_nothing(self):
        # Arrange - invalid input should not raise; UPDATE matches no rows.
        await self.store.save_run(a_summary())
        await self.store.update_model_metrics(999, {"a": 1})
        self.assertEqual(
            self.rows(f"SELECT COUNT(*) FROM {MODEL_TABLE}"
                      f" WHERE data_quality_metrics IS NOT NULL")[0][0], 0)


if __name__ == "__main__":
    unittest.main()


class TestTheTunerRecordsItsOwnRun(unittest.IsolatedAsyncioTestCase):
    """End to end: a real run writes its own rows, no host code involved."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.path = os.path.join(self.directory, "training.sqlite")

    async def run_tuner(self, **kwargs):
        from unittest.mock import patch
        from ezmt.hyperparameters import DiscreteNonOrdinal
        from ezmt.model_tuner import ModelTuner
        from tests.unit.sequential_unit_test import build_model_space

        options = {"sequential": True, "goal": "max", "directory": self.directory,
                   "temp_directory": f"{self.directory}/.tmp",
                   "database_path": self.path}
        options.update(kwargs)
        tuner = ModelTuner(build_model_space(),
                           {"mode": DiscreteNonOrdinal(["a", "b"])},
                           pop_size=4, **options)
        with patch("ezmt.model_tuner.choose_hyperparams",
                   side_effect=[{"mode": m} for m in "aabb"] * 4):
            return await tuner.run("test")

    def rows(self, sql):
        connection = sqlite3.connect(self.path)
        try:
            return connection.execute(sql).fetchall()
        finally:
            connection.close()

    async def test_a_run_writes_its_population(self):
        # Arrange / Act
        await self.run_tuner()
        # Assert
        self.assertEqual(self.rows(f"SELECT COUNT(*) FROM {MODEL_TABLE}")[0][0], 4)

    async def test_the_summary_carries_the_ids_back(self):
        # Arrange - a host attaches its own metrics using them.
        _best, summary = await self.run_tuner()
        # Assert
        self.assertIn("training_run_id", summary)
        self.assertIn("winner_model_id", summary)

    async def test_the_recorded_winner_matches_the_returned_organism(self):
        _best, summary = await self.run_tuner()
        recorded = self.rows(
            f"SELECT score FROM {MODEL_TABLE} WHERE is_winner = 1")[0][0]
        self.assertEqual(recorded, summary["best"]["score"])

    async def test_notes_reach_the_run_row(self):
        await self.run_tuner(notes="a note")
        self.assertEqual(self.rows(f"SELECT notes FROM {RUN_TABLE}")[0][0], "a note")

    async def test_without_a_database_path_nothing_is_recorded(self):
        # Arrange - recording is opt-in; ezmt was a pure library before this.
        _best, summary = await self.run_tuner(database_path=None)
        # Assert
        self.assertFalse(os.path.exists(self.path))
        self.assertNotIn("training_run_id", summary)
