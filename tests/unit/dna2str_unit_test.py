import unittest

from ezmt.organism import dna2str


def gene(name, args=None, kwargs=None):
    return {
        "name": name,
        "train": {"func": lambda x: x, "args": args or [], "kwargs": kwargs or {}},
    }


class TestDna2StrCacheIdentity(unittest.TestCase):
    """dna2str is the key for ezmt's per-gene result cache.

    Organisms share a cached result when their strings match up to that gene, so
    the string must capture whatever makes their computations differ. Genes store
    hyperparameters by name, so without the sampled values two organisms with
    different hyperparameters render identically, collapse onto one cache entry,
    and only the first ever runs.
    """

    def test_differing_hyperparameters_produce_different_keys(self):
        # Arrange
        dna = [gene("build", kwargs={"width": "hidden_dim"})]
        # Act
        small = dna2str(dna, {"hidden_dim": 250})
        large = dna2str(dna, {"hidden_dim": 750})
        # Assert
        self.assertNotEqual(small, large)

    def test_identical_hyperparameters_produce_identical_keys(self):
        # Arrange - this is what preserves prefix sharing for the shared pipeline.
        dna = [gene("load", kwargs={"window": "max_window"})]
        # Act / Assert
        self.assertEqual(
            dna2str(dna, {"max_window": 24}), dna2str(dna, {"max_window": 24})
        )

    def test_genes_consuming_no_hyperparameters_still_share(self):
        # Arrange
        dna = [gene("split", kwargs={"ratio": 0.8})]
        # Act / Assert
        self.assertEqual(dna2str(dna, {"lr": 1e-3}), dna2str(dna, {"lr": 1e-5}))

    def test_hyperparameters_resolve_in_positional_args_too(self):
        # Arrange
        dna = [gene("train", args=["learning_rate"])]
        # Act / Assert
        self.assertNotEqual(dna2str(dna, {"learning_rate": 1e-3}),
                            dna2str(dna, {"learning_rate": 1e-5}))

    def test_unknown_strings_are_left_alone(self):
        # Arrange - a plain string arg is data, not a hyperparameter reference.
        dna = [gene("load", kwargs={"mode": "hour"})]
        # Act / Assert
        self.assertIn("mode=hour", dna2str(dna, {"hidden_dim": 250}))

    def test_prefixes_diverge_only_at_the_consuming_gene(self):
        # Arrange - the property the cache depends on: a shared pipeline stays
        # shared, and organisms fork only once a differing hyperparameter is used.
        dna = [
            gene("load", kwargs={"window": "max_window"}),
            gene("build", kwargs={"width": "hidden_dim"}),
        ]
        a = {"max_window": 24, "hidden_dim": 250}
        b = {"max_window": 24, "hidden_dim": 750}
        # Act / Assert
        self.assertEqual(dna2str(dna[:1], a), dna2str(dna[:1], b))
        self.assertNotEqual(dna2str(dna[:2], a), dna2str(dna[:2], b))

    def test_omitting_parameters_renders_references_unresolved(self):
        # Arrange - the display path; callers keying a cache must pass parameters.
        dna = [gene("build", kwargs={"width": "hidden_dim"})]
        # Act / Assert
        self.assertIn("width=hidden_dim", dna2str(dna))


if __name__ == "__main__":
    unittest.main()
