import unittest

from ezmt.hyperparameters import ContinuousRange, DiscreteNonOrdinal, DiscreteOrdinal


class TestContinuousRange(unittest.TestCase):

    def test_sample_stays_inside_the_range(self):
        hp = ContinuousRange(0.0, 1.0)
        for _ in range(100):
            self.assertGreaterEqual(hp.sample(), 0.0)
            self.assertLessEqual(hp.sample(), 1.0)

    def test_mutation_is_clamped_to_the_range(self):
        # Arrange - a mutation step wider than the range would otherwise walk out
        # of it from either end.
        hp = ContinuousRange(0.0, 1.0, max_change_perc=10.0)
        # Act
        low = [hp.mutate(0.0) for _ in range(50)]
        high = [hp.mutate(1.0) for _ in range(50)]
        # Assert
        self.assertTrue(all(0.0 <= v <= 1.0 for v in low + high))

    def test_start_must_be_below_end(self):
        with self.assertRaises(ValueError):
            ContinuousRange(1.0, 1.0)

    def test_non_numeric_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            ContinuousRange("a", "b")


class TestDiscreteOrdinal(unittest.TestCase):
    """Ordinal mutation walks the option list by index, so it must stay in bounds."""

    def test_mutation_moves_at_most_one_index(self):
        hp = DiscreteOrdinal([0, 1, 2, 3, 4])
        for _ in range(50):
            self.assertIn(hp.mutate(2), (1, 2, 3))

    def test_mutation_at_the_edges_is_clamped(self):
        # Arrange - index 0 - 1 and index n-1 + 1 are both out of range.
        hp = DiscreteOrdinal([10, 20, 30])
        # Act / Assert
        self.assertIn(hp.mutate(10), (10, 20))
        self.assertIn(hp.mutate(30), (20, 30))

    def test_a_single_option_always_mutates_to_itself(self):
        # Arrange - pinning a gene to one value is how the search holds an axis
        # fixed, so this is the common case, not a degenerate one.
        hp = DiscreteOrdinal([7])
        # Act / Assert
        for _ in range(20):
            self.assertEqual(hp.mutate(7), 7)

    def test_a_value_outside_the_options_is_rejected(self):
        hp = DiscreteOrdinal([1, 2, 3])
        with self.assertRaises(ValueError):
            hp.mutate(99)

    def test_a_string_is_not_a_valid_option_list(self):
        with self.assertRaises(ValueError):
            DiscreteOrdinal("abc")


class TestDiscreteNonOrdinal(unittest.TestCase):
    """Non-ordinal options have no neighbours, so mutation picks any other one."""

    def test_mutation_always_changes_the_value(self):
        hp = DiscreteNonOrdinal(["a", "b", "c"])
        for _ in range(50):
            self.assertNotEqual(hp.mutate("a"), "a")

    def test_mutation_can_reach_every_other_option(self):
        hp = DiscreteNonOrdinal(["a", "b", "c"])
        self.assertEqual({hp.mutate("a") for _ in range(100)}, {"b", "c"})

    def test_a_single_option_returns_itself_instead_of_hanging(self):
        # Arrange - the old implementation resampled until it saw a new value,
        # which never terminates here. A pinned non-ordinal gene froze the whole
        # run at the first mutation, with no error to point at.
        hp = DiscreteNonOrdinal(["only"])
        # Act / Assert
        self.assertEqual(hp.mutate("only"), "only")

    def test_duplicate_options_do_not_hang(self):
        # Arrange - len(options) > 1 is not enough; what matters is whether any
        # option actually differs from the current value.
        hp = DiscreteNonOrdinal(["same", "same"])
        # Act / Assert
        self.assertEqual(hp.mutate("same"), "same")

    def test_a_set_is_not_a_valid_option_list(self):
        with self.assertRaises(ValueError):
            DiscreteNonOrdinal({"a", "b"})


if __name__ == "__main__":
    unittest.main()
