import unittest
from EZMT import ModelTuner
from organism import Organism
from config_validation import ContinuousRange
import pandas as pd


class TestModelTunerPopulationInitialization(unittest.TestCase):

    def setUp(self):
        # Sample data and model space setup for testing
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
            'label': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        model_space = [
            [
                {'name': 'gene1',
                 'train': {'func': lambda x: x, 'inputs': 'x_train', 'outputs': 'output1', 'args': [], 'kwargs': {}}},
                {'name': 'gene2',
                 'train': {'func': lambda x: x ** 2, 'inputs': 'output1', 'outputs': 'output2', 'args': [],
                           'kwargs': {}}}
            ],
            [
                {'name': 'gene3',
                 'train': {'func': lambda x, y: x + y + 1, 'inputs': 'output2', 'outputs': 'output3',
                           'args': [ContinuousRange(0, 10)], 'kwargs': {}}}
            ]
        ]
        self.model_tuner = ModelTuner(model_space, data, y_col='label', pop_size=20)

    def test_population_size(self):
        """Test that population is initialized with the correct size"""
        self.model_tuner.populate_init()
        self.assertEqual(len(self.model_tuner.population), 20, "Population size does not match specified pop_size.")

    def test_gene_length(self):
        """Test that each organism in the population is initialized with the correct gene structure"""
        self.model_tuner.populate_init()
        for organism in self.model_tuner.population:
            self.assertIsInstance(organism, Organism, "Population should contain Organism instances.")
            self.assertEqual(len(organism.dna), len(self.model_tuner.model_space),
                             "Each organism's DNA length should match the model space length.")

    def test_gene_options(self):
        self.model_tuner.populate_init()
        for organism in self.model_tuner.population:
            for gene, gene_space in zip(organism.dna, self.model_tuner.model_space):
                self.assertIn(gene['name'], [gs['name'] for gs in gene_space],
                              "Organism gene names should match options in model space.")

    def test_population_diversity(self):
        """Test that the initial population has varied genes based on model space options"""
        self.model_tuner.populate_init()
        # Collect DNA strings to check for diversity in initial population
        dna_strings = {str(organism.dna) for organism in self.model_tuner.population}
        # A population with high diversity should have multiple unique DNA sequences
        self.assertEqual(len(dna_strings), 20, "Population appears to lack diversity.")

    def test_gene_diversity(self):
        """Test that each gene + arg combination has a variety of choices in the population"""
        self.model_tuner.populate_init()
        # Collect DNA strings to check for diversity in initial population
        for i in range(len(self.model_tuner.model_space)):
            gene_pool = {str(organism.dna[i]) for organism in self.model_tuner.population}
            self.assertGreater(len(gene_pool), 1, "Each organism's gene options should be diverse.")

    def test_gene_initialization_with_random_choices(self):
        """Test that all gene names are represented in the initial population"""
        self.model_tuner.populate_init()
        gene_counts = {gene['name']: 0 for gene_space in self.model_tuner.model_space for gene in gene_space}

        for organism in self.model_tuner.population:
            for gene in organism.dna:
                gene_counts[gene['name']] += 1

        # Verify that each gene from the model space appears at least once across the population
        for gene_name, count in gene_counts.items():
            self.assertGreater(count, 0, f"Gene '{gene_name}' was never chosen in the initial population.")


if __name__ == '__main__':
    unittest.main()
