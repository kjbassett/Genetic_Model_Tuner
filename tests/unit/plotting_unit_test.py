"""The run's score chart.

The chart is the report on whether selection improved anything, so it has to
show every organism -- including the ones that scored identically -- and it must
never be the reason a finished run fails.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ezmt.plotting import GENERATION_SCORES_FILE, plot_generation_scores, spread


def history(scores_per_generation):
    """Build a generation_history from plain score lists."""
    return [
        [
            {"generation": n + 1, "organism_index": i, "score": score}
            for i, score in enumerate(scores)
        ]
        for n, scores in enumerate(scores_per_generation)
    ]


class PlottingTestCase(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def image(self, name=GENERATION_SCORES_FILE):
        return os.path.join(self.folder, name)


class TestSpread(unittest.TestCase):

    def test_a_lone_point_is_not_moved(self):
        self.assertEqual(spread(1), [0.0])

    def test_points_are_centred_on_the_generation(self):
        # Arrange - offsets are added to the generation number, so a lopsided
        # spread would put a generation's cloud beside its own tick.
        offsets = spread(4)
        # Assert
        self.assertAlmostEqual(sum(offsets), 0.0)

    def test_no_two_organisms_share_an_offset(self):
        # Arrange - this is the whole point: an elite carried forward scores
        # identically to itself, and would otherwise plot underneath itself.
        offsets = spread(8)
        self.assertEqual(len(set(offsets)), 8)

    def test_offsets_stay_inside_the_width(self):
        # Arrange - wider than 1.0 and adjacent generations would overlap.
        offsets = spread(8, width=0.6)
        self.assertLessEqual(max(offsets) - min(offsets), 0.6)


class TestPlotGenerationScores(PlottingTestCase):

    def test_it_writes_the_image(self):
        # Arrange / Act
        path = plot_generation_scores(history([[0.1, 0.2], [0.3, 0.15]]),
                                      self.folder)
        # Assert
        self.assertEqual(path, self.image())
        self.assertGreater(os.path.getsize(path), 0)

    def test_it_creates_the_folder_it_is_given(self):
        # Arrange - it is called before anything else has written there.
        folder = os.path.join(self.folder, "run", "version")
        # Act
        plot_generation_scores(history([[0.1]]), folder)
        # Assert
        self.assertTrue(os.path.isfile(os.path.join(folder,
                                                    GENERATION_SCORES_FILE)))

    def test_nothing_is_drawn_for_an_empty_run(self):
        self.assertIsNone(plot_generation_scores([], self.folder))
        self.assertFalse(os.path.exists(self.image()))

    def test_organisms_without_a_score_are_skipped(self):
        # Arrange - an organism can fail before it is scored, and None would
        # raise inside matplotlib rather than here.
        path = plot_generation_scores(history([[0.1, None], [None, None]]),
                                      self.folder)
        # Assert - the one scored organism is still worth a chart
        self.assertIsNotNone(path)

    def test_a_run_with_no_scores_at_all_draws_nothing(self):
        self.assertIsNone(
            plot_generation_scores(history([[None, None]]), self.folder))

    def test_a_missing_matplotlib_is_not_an_error(self):
        # Arrange - it is a report, not a result. An environment without it
        # should still be able to finish a run.
        with patch.dict("sys.modules", {"matplotlib": None}):
            path = plot_generation_scores(history([[0.1]]), self.folder)
        # Assert
        self.assertIsNone(path)

    def test_it_handles_a_single_generation_of_one(self):
        # Arrange - run_short_genetic_algorithm's shape: one organism, one
        # generation, so the best-of line has a single point and no segment.
        path = plot_generation_scores(history([[0.42]]), self.folder)
        self.assertIsNotNone(path)

    def test_a_minimising_run_lines_up_the_lowest(self):
        # Arrange - goal is not decoration; the "best of generation" line has to
        # follow the same end the tuner selects on.
        path = plot_generation_scores(
            history([[0.5, -0.5]]), self.folder, goal="min")
        self.assertIsNotNone(path)


if __name__ == "__main__":
    unittest.main()
