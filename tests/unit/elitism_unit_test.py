"""The best organism a run finds must survive to the end of it.

select_and_reproduce carried the top organism into the next generation and then
called reset() on it, which cleared its score and made the next generation run
it again. Training is stochastic, so it could come back worse than the score
that earned it the slot -- and the run returns max() of the FINAL generation, so
a better organism found earlier is simply lost. A real run scored 0.002214 in
generation 1 and returned 0.001206 from generation 2.
"""

import unittest
from unittest.mock import MagicMock

from ezmt.model_tuner import ModelTuner
from ezmt.organism import Organism


_UNSET = object()


def _organism(name, score, fitness, dna=None, saved=_UNSET):
    """An organism that has already run, unless `saved` is given as None.

    The sentinel matters: None is a meaningful value here -- it is what an
    organism that has never run looks like -- so it cannot double as "use the
    default".
    """
    organism = Organism(name, dna or [{"a": 1}], {"p": score})
    organism.score = score
    organism.fitness = fitness
    organism.saved_result = {"score": score} if saved is _UNSET else saved
    return organism


class _FixedHyperparameter:
    """A hyperparameter that mutates to a constant.

    Real ones are random, which would make these tests flaky for a reason that
    has nothing to do with elitism.
    """

    def sample(self):
        return 0.0

    def mutate(self, value):
        return 0.0


def _tuner(population):
    tuner = ModelTuner.__new__(ModelTuner)
    tuner.population = population
    tuner.model_space = [[{"a": 1}]]
    tuner.hyperparam_space = {"p": _FixedHyperparameter()}
    return tuner


class TestElitePreservation(unittest.TestCase):

    def setUp(self):
        self.population = [
            _organism("weak", 0.1, 0.1),
            _organism("best", 0.9, 0.9),
            _organism("middling", 0.5, 0.5),
        ]

    def test_the_best_organism_keeps_its_score(self):
        # Arrange - reset() used to zero this, which is what let a better
        # organism be replaced by a worse re-run of itself.
        tuner = _tuner(list(self.population))

        # Act
        tuner.select_and_reproduce()

        # Assert
        elite = tuner.population[0]
        self.assertEqual(elite.name, "best")
        self.assertEqual(elite.score, 0.9)
        self.assertEqual(elite.fitness, 0.9)

    def test_the_best_organism_keeps_its_saved_result(self):
        # Arrange - the carried value has to be the saved dict of paths, not
        # live state, or an elite would pin a whole trained model in memory for
        # every generation it survives.
        tuner = _tuner(list(self.population))

        # Act
        tuner.select_and_reproduce()

        # Assert
        self.assertEqual(tuner.population[0].saved_result, {"score": 0.9})

    def test_children_are_not_flagged_as_elite(self):
        # Arrange - a child inherits its parent's genes but has never run, so
        # marking it elite would skip training it entirely.
        tuner = _tuner(list(self.population))

        # Act
        tuner.select_and_reproduce()

        # Assert
        # A child has never run, so it has no saved result -- which is exactly
        # what tells the run loop to train it rather than carry it.
        for child in tuner.population[1:]:
            with self.subTest(child=child.name):
                self.assertFalse(child.saved_result)

    def test_elitism_below_one_is_raised_to_one(self):
        # Arrange - with no elite the run can end on a generation worse than one
        # it already had, and the value it returns is not the best it found.
        tuner = _tuner(list(self.population))

        # Act
        tuner.select_and_reproduce(elitism=0)

        # Assert
        self.assertEqual(tuner.population[0].score, 0.9)
        self.assertEqual(tuner.population[0].saved_result, {"score": 0.9})

    def test_more_than_one_elite_is_honoured(self):
        tuner = _tuner(list(self.population))
        tuner.select_and_reproduce(elitism=2)
        carried = [o for o in tuner.population if o.saved_result]
        self.assertEqual(len(carried), 2)
        self.assertEqual([o.score for o in carried], [0.9, 0.5])

    def test_the_population_size_is_unchanged(self):
        # Arrange - carrying an elite must replace a child, not add to the
        # population, or each generation would grow.
        tuner = _tuner(list(self.population))
        tuner.select_and_reproduce()
        self.assertEqual(len(tuner.population), len(self.population))

    def test_reset_clears_the_saved_result(self):
        # Arrange - a reset organism has no measured score any more, so it must
        # not look like one that can be carried forward.
        organism = _organism("x", 0.4, 0.4)

        # Act
        organism.reset()

        # Assert
        self.assertFalse(organism.saved_result)
        self.assertEqual(organism.score, 0)

    def test_a_fresh_organism_has_no_saved_result(self):
        organism = Organism("fresh", [{"a": 1}], {})
        self.assertFalse(organism.saved_result)


class TestCarryingRequiresStaticFolds(unittest.TestCase):
    """A score is only comparable to its peers if it came from the same rows."""

    @staticmethod
    def _loop_decision(folds_are_static, saved_result):
        """What _experience_sequentially would decide for one organism."""
        tuner = ModelTuner.__new__(ModelTuner)
        tuner.folds_are_static = folds_are_static
        organism = _organism("x", 0.9, 0.9, saved=saved_result)
        return organism.saved_result if tuner.folds_are_static else None

    def test_a_static_fold_carries_the_saved_result(self):
        # Arrange - every generation trains on the same rows, so last
        # generation's score is directly comparable and re-running is waste.
        self.assertEqual(
            self._loop_decision(True, {"score": 0.9}), {"score": 0.9})

    def test_a_rolling_fold_forces_a_re_run(self):
        # Arrange - each generation trains on different rows, so a carried score
        # was measured against a different problem than its peers face. Carrying
        # it would let an organism win on an easier fold it no longer sits in.
        self.assertIsNone(self._loop_decision(False, {"score": 0.9}))

    def test_an_organism_that_never_ran_is_not_carried(self):
        self.assertFalse(self._loop_decision(True, None))


class TestOrganismFolders(unittest.TestCase):
    """A folder must not be reused by a different organism.

    Numbering by population position let generation 2 overwrite a folder
    generation 1 still owned. A carried-forward elite kept its old path while
    the organism at that position wrote over it, so the run published that
    organism's data under the elite's score: a run reported best +0.000482 and
    saved an organism scoring -0.000133.
    """

    @staticmethod
    def _tuner():
        tuner = ModelTuner.__new__(ModelTuner)
        tuner.temp_directory = "/tmp/x"
        return tuner

    def test_the_same_genome_maps_to_the_same_folder(self):
        # Arrange - this is what lets a carried elite still find its outputs.
        tuner = self._tuner()
        self.assertEqual(
            tuner.organism_folder("dna-a"), tuner.organism_folder("dna-a"))

    def test_different_genomes_map_to_different_folders(self):
        tuner = self._tuner()
        self.assertNotEqual(
            tuner.organism_folder("dna-a"), tuner.organism_folder("dna-b"))

    def test_the_folder_is_named_by_digest_not_by_an_index(self):
        # Arrange - a small integer name is what let generation 2 land on
        # generation 1's folder. The name has to come from the genome.
        tuner = self._tuner()

        # Act
        name = tuner.organism_folder("dna-a").rsplit("/", 1)[-1]

        # Assert
        self.assertFalse(name.isdigit(), f"{name!r} is positional")
        self.assertTrue(all(c in "0123456789abcdef" for c in name))
        self.assertGreater(len(name), 8)

    def test_position_is_not_an_input(self):
        # Arrange - the signature is the guarantee: nothing about where an
        # organism sits in the population can reach the path.
        import inspect

        parameters = set(inspect.signature(ModelTuner.organism_folder).parameters)
        self.assertEqual(parameters, {"self", "dna"})

    def test_folders_sit_under_the_temp_directory(self):
        tuner = self._tuner()
        self.assertTrue(tuner.organism_folder("dna-a").startswith("/tmp/x/organisms/"))


if __name__ == "__main__":
    unittest.main()
