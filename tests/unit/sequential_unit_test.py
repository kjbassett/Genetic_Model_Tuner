import os
import shutil
import tempfile
import unittest
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import pandas as pd

from ezmt.hyperparameters import DiscreteNonOrdinal, DiscreteOrdinal
from ezmt.model_tuner import (
    CHECKPOINT_MARKER_FILE,
    CHECKPOINT_STATE_FILE,
    ModelTuner,
    find_checkpoint_prefixes,
)
from ezmt.organism import Organism, dna2str

# Counts how many times each gene actually ran. Sequential genes stay in this
# process, so a module-level counter sees every call.
GENE_CALLS = Counter()


def load_rows():
    GENE_CALLS["load_rows"] += 1
    return pd.DataFrame({"value": [1.0, 2.0, 3.0]})


def double_rows(frame):
    GENE_CALLS["double_rows"] += 1
    return frame * 2


def score_rows(frame, mode):
    GENE_CALLS["score_rows"] += 1
    return float(frame["value"].sum()) + (1.0 if mode == "a" else 2.0)


def build_model_space():
    """Three genes where only the last one reads a hyperparameter.

    The population therefore shares one prefix through gene 1 and forks at gene
    2, which is the shape the checkpoint logic exists for.
    """
    return [
        {"name": "load_rows", "train": {"func": load_rows, "outputs": ["frame"]}},
        {
            "name": "double_rows",
            "train": {"func": double_rows, "args": ["frame"], "outputs": ["frame"]},
        },
        {
            "name": "score_rows",
            "train": {
                "func": score_rows,
                "args": ["frame", "mode"],
                "outputs": ["score"],
            },
        },
    ]


class SequentialTestCase(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        GENE_CALLS.clear()
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def build_tuner(self, modes=("a", "a", "b", "b"), **kwargs):
        """A tuner whose population has exactly the given hyperparameter values.

        populate_init samples at random, which would make the number of branches
        -- the thing every test here is about -- vary run to run.
        """
        options = {"sequential": True, "goal": "max", "directory": self.directory}
        options.update(kwargs)
        tuner = ModelTuner(
            build_model_space(),
            {"mode": DiscreteNonOrdinal(["a", "b"])},
            pop_size=len(modes),
            **options,
        )
        tuner.populate_init("test")
        for organism, mode in zip(tuner.population, modes):
            organism.parameters["mode"] = mode
        return tuner


class TestFindCheckpointPrefixes(SequentialTestCase):
    """Only prefixes the population actually forks after are worth writing."""

    def test_the_prefix_before_the_fork_is_the_only_checkpoint(self):
        # Arrange
        tuner = self.build_tuner()
        # Act
        checkpoints = find_checkpoint_prefixes(tuner.population)
        # Assert - genes 0 and 1 render identically for everyone, so the split
        # happens after gene 1 and only that prefix is worth caching.
        organism = tuner.population[0]
        expected = dna2str(organism.dna[:2], organism.parameters)
        self.assertEqual(checkpoints, {expected: 4})

    def test_a_population_that_never_forks_has_no_checkpoints(self):
        # Arrange - identical organisms share every prefix, so nothing is ever
        # read back and writing a checkpoint would be pure cost.
        tuner = self.build_tuner(modes=("a", "a"))
        # Act / Assert
        self.assertEqual(find_checkpoint_prefixes(tuner.population), {})

    def test_a_single_organism_has_no_checkpoints(self):
        tuner = self.build_tuner(modes=("a",))
        self.assertEqual(find_checkpoint_prefixes(tuner.population), {})

    def test_an_empty_population_has_no_checkpoints(self):
        self.assertEqual(find_checkpoint_prefixes([]), {})

    def test_the_last_gene_is_never_a_checkpoint(self):
        # Arrange - nothing runs after it, so its state would never be read.
        tuner = self.build_tuner()
        organism = tuner.population[0]
        full = dna2str(organism.dna, organism.parameters)
        # Act
        checkpoints = find_checkpoint_prefixes(tuner.population)
        # Assert
        self.assertNotIn(full, checkpoints)


class TestSharedWorkRunsOnce(SequentialTestCase):

    async def test_genes_before_the_fork_run_once_for_the_whole_population(self):
        # Arrange
        tuner = self.build_tuner()
        # Act
        await tuner.experience_population({}, None)
        # Assert - the first organism computes the shared prefix, the other three
        # read it back off disk.
        self.assertEqual(GENE_CALLS["load_rows"], 1)
        self.assertEqual(GENE_CALLS["double_rows"], 1)

    async def test_identical_genomes_are_only_run_once(self):
        # Arrange - two organisms per mode, so half the population is duplicate.
        tuner = self.build_tuner()
        # Act
        results = await tuner.experience_population({}, None)
        # Assert
        self.assertEqual(GENE_CALLS["score_rows"], 2)
        self.assertEqual(len(results), 2)

    async def test_every_organism_gets_a_result_including_the_duplicates(self):
        # Arrange - score_fitness looks each organism's DNA up in this dict, so a
        # skipped duplicate still has to find its entry.
        tuner = self.build_tuner()
        # Act
        results = await tuner.experience_population({}, None)
        # Assert
        for organism in tuner.population:
            self.assertIn(dna2str(organism.dna, organism.parameters), results)

    async def test_results_hold_scores_rather_than_objects(self):
        # Arrange - score_fitness reads data["score"], and the point of releasing
        # the state is that nothing heavy stays reachable through these results.
        tuner = self.build_tuner()
        # Act
        results = await tuner.experience_population({}, None)
        # Assert
        for saved in results.values():
            self.assertIsInstance(saved["score"], float)
            self.assertIsInstance(saved["frame"], str)

    async def test_organisms_release_their_state_after_saving(self):
        tuner = self.build_tuner()
        await tuner.experience_population({}, None)
        for organism in tuner.population:
            self.assertEqual(organism.knowledge, {})


class TestCheckpointRestore(SequentialTestCase):

    async def test_the_checkpoint_is_written_where_the_population_forks(self):
        # Arrange
        tuner = self.build_tuner()
        # Act
        await tuner.experience_population({}, None)
        # Assert
        prefix = next(iter(find_checkpoint_prefixes(tuner.population)))
        folder = tuner.checkpoint_folder(prefix)
        self.assertTrue(os.path.exists(os.path.join(folder, CHECKPOINT_STATE_FILE)))
        self.assertTrue(os.path.exists(os.path.join(folder, CHECKPOINT_MARKER_FILE)))

    async def test_a_resumed_state_matches_what_the_first_organism_had(self):
        # Arrange
        tuner = self.build_tuner()
        await tuner.experience_population({}, None)
        # Act
        state, first_gene = tuner.restore_from_checkpoint(tuner.population[1], {})
        # Assert - load_rows then double_rows, so 1/2/3 doubled
        self.assertEqual(first_gene, 2)
        self.assertEqual(list(state["frame"]["value"]), [2.0, 4.0, 6.0])

    async def test_two_organisms_resuming_get_independent_objects(self):
        # Arrange - without a process pool nothing pickles the state between
        # genes, so a shared object would carry one organism's in-place edits
        # into the next. This is the regression guard for that.
        tuner = self.build_tuner()
        await tuner.experience_population({}, None)
        # Act
        first, _ = tuner.restore_from_checkpoint(tuner.population[1], {})
        first["frame"].loc[0, "value"] = 999.0
        second, _ = tuner.restore_from_checkpoint(tuner.population[2], {})
        # Assert
        self.assertEqual(second["frame"].loc[0, "value"], 2.0)

    def test_an_organism_with_no_cached_prefix_starts_from_the_base_state(self):
        # Arrange
        tuner = self.build_tuner()
        base = {"seed": 1}
        # Act
        state, first_gene = tuner.restore_from_checkpoint(tuner.population[0], base)
        # Assert
        self.assertEqual(first_gene, 0)
        self.assertEqual(state, base)
        self.assertIsNot(state, base)

    async def test_a_checkpoint_without_its_marker_is_ignored(self):
        # Arrange - the marker is written last, so a folder missing it is a save
        # that was interrupted and must not be loaded as a truncated state.
        tuner = self.build_tuner()
        await tuner.experience_population({}, None)
        prefix = next(iter(find_checkpoint_prefixes(tuner.population)))
        os.remove(os.path.join(tuner.checkpoint_folder(prefix), CHECKPOINT_MARKER_FILE))
        # Act
        _, first_gene = tuner.restore_from_checkpoint(tuner.population[1], {})
        # Assert
        self.assertEqual(first_gene, 0)

    async def test_a_different_fold_does_not_reuse_a_cached_prefix(self):
        # Arrange - when generations train on different rows, a prefix cached in
        # one is wrong for the next even though the DNA is identical.
        tuner = self.build_tuner()
        await tuner.experience_population({}, None)
        tuner.fold_key = "1"
        # Act
        _, first_gene = tuner.restore_from_checkpoint(tuner.population[1], {})
        # Assert
        self.assertEqual(first_gene, 0)


class TestNoPoolInSequentialMode(SequentialTestCase):

    async def test_a_sequential_run_never_builds_a_process_pool(self):
        # Arrange - a pool would add worker processes and force every sync gene
        # to pickle the whole state across a process boundary, which is the cost
        # sequential mode exists to avoid.
        tuner = self.build_tuner()
        # Act
        with patch("ezmt.model_tuner.ProcessPoolExecutor") as pool_cls:
            await tuner.run("test")
        # Assert
        pool_cls.assert_not_called()

    async def test_a_concurrent_run_still_builds_a_process_pool(self):
        # Arrange - a spy rather than a stand-in, because the concurrent path
        # then goes on to submit real work to whatever it was handed.
        tuner = self.build_tuner(sequential=False)
        # Act
        with patch(
            "ezmt.model_tuner.ProcessPoolExecutor", side_effect=ProcessPoolExecutor
        ) as pool_cls:
            await tuner.run("test")
        # Assert
        pool_cls.assert_called_once()

    async def test_sequential_and_concurrent_runs_agree_on_the_scores(self):
        # Arrange - the two paths must be interchangeable, not merely both able
        # to finish.
        sequential = self.build_tuner()
        concurrent = self.build_tuner(sequential=False)
        # Act
        sequential_scores = {
            dna: saved["score"]
            for dna, saved in (
                await sequential.experience_population({}, None)
            ).items()
        }
        concurrent_results = await concurrent.experience_population({}, None)
        # Assert
        self.assertEqual(
            sequential_scores,
            {dna: state["score"] for dna, state in concurrent_results.items()},
        )


class TestPublishingTheBestOrganism(SequentialTestCase):
    """run() leaves the winner on disk and readable, on both paths."""

    async def _run(self, **kwargs):
        tuner = self.build_tuner(**kwargs)
        best = await tuner.run("test")
        return tuner, best

    async def test_the_winner_lands_in_the_normal_run_folder(self):
        # Arrange / Act
        tuner, best = await self._run()
        # Assert - not the numbered temp folder the loop used
        self.assertTrue(best.folder.startswith(f"{self.directory}/test/"))
        self.assertNotIn(tuner.temp_directory, best.folder)
        self.assertTrue(os.path.exists(f"{best.folder}/knowledge.json"))

    async def test_the_winners_knowledge_comes_back_as_real_objects(self):
        # Arrange - callers read predictions and metrics straight off knowledge,
        # so a file name in place of a frame is not good enough.
        _, best = await self._run()
        # Assert
        self.assertIsInstance(best.knowledge["frame"], pd.DataFrame)

    async def test_the_concurrent_path_also_saves_the_winner(self):
        # Arrange - this used to be the caller's job, which meant every caller
        # had to remember to do it.
        _, best = await self._run(sequential=False)
        # Assert
        self.assertTrue(os.path.exists(f"{best.folder}/knowledge.json"))
        self.assertIsInstance(best.knowledge["frame"], pd.DataFrame)

    async def test_the_winner_is_the_highest_scoring_organism(self):
        # Arrange - mode "b" scores 2.0 higher than mode "a"
        _, best = await self._run(modes=("a", "a", "b", "b"))
        # Assert
        self.assertEqual(best.parameters["mode"], "b")

    async def test_cleanup_temp_removes_the_checkpoint_tree(self):
        # Arrange / Act
        tuner, best = await self._run(cleanup_temp=True)
        # Assert - and the published winner must survive the cleanup
        self.assertFalse(os.path.exists(tuner.temp_directory))
        self.assertTrue(os.path.exists(f"{best.folder}/knowledge.json"))

    async def test_the_temp_tree_is_kept_by_default(self):
        tuner, _ = await self._run()
        self.assertTrue(os.path.exists(tuner.temp_directory))


class TestSequentialConfiguration(unittest.TestCase):

    def test_an_unknown_save_organisms_value_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelTuner(build_model_space(), {"mode": DiscreteOrdinal(["a"])},
                       save_organisms="some")

    def test_folds_are_static_when_no_data_is_supplied(self):
        # Arrange - with no data every generation gets the same empty fold, so a
        # prefix cached in one generation is still valid in the next.
        tuner = ModelTuner(build_model_space(), {"mode": DiscreteOrdinal(["a"])})
        # Act / Assert
        self.assertTrue(tuner.folds_are_static)

    def test_folds_are_not_static_when_data_is_supplied(self):
        # Arrange
        data = pd.DataFrame({"f": range(10), "label": [0, 1] * 5})
        # Act
        tuner = ModelTuner(
            build_model_space(), {"mode": DiscreteOrdinal(["a"])},
            data=data, y_col="label",
        )
        # Assert
        self.assertFalse(tuner.folds_are_static)


if __name__ == "__main__":
    unittest.main()
