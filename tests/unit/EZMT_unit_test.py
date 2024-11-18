import unittest
from EZMT import ModelTuner, dna2str
from organism import Organism
from config_validation import ContinuousRange
import pandas as pd
import numpy as np
from scipy.stats import binom


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


class TestModelTunerGoals(unittest.TestCase):

    def setUp(self):
        # Sample data and model space setup for testing
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
            'label': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        model_space = [
            [{
                'name': 'gene1',
                'train': {
                    'func': lambda x: x,
                    'inputs': 'x_train',
                    'outputs': 'output',
                    'args': [ContinuousRange(0, 1)], 'kwargs': {}
                }
            }]
        ]
        self.model_tuner_min = ModelTuner(model_space, data, y_col='label', goal='min')
        self.model_tuner_max = ModelTuner(model_space, data, y_col='label', goal='max')

        # Populate the initial population for testing
        self.model_tuner_min.populate_init()
        self.model_tuner_max.populate_init()

    def test_fitness_score_range_min(self):
        """Test that the fitness scoring logic works correctly for minimization"""
        # Assign unique scores to organisms based on their DNA
        unique_organisms = {
            dna2str(organism.dna): {'score': score}
            for organism, score in zip(self.model_tuner_min.population, range(self.model_tuner_min.population_size))
        }

        # Score the fitness
        self.model_tuner_min.score_fitness(unique_organisms)
        fitness_scores = [model.fitness for model in self.model_tuner_min.population]

        # Check that fitness scores are correctly assigned (highest fitness for the lowest score)
        self.assertEqual(max(fitness_scores), 1.0, "Best fitness score should be 1.0 for the lowest score.")
        self.assertEqual(min(fitness_scores), 0.0, "Worst fitness score should be 0.0 for the highest score.")


    def test_fitness_score_range_max(self):
        """Test that the fitness scoring logic works correctly for maximization"""
        # Assign unique scores to organisms based on their DNA
        unique_organisms = {
            dna2str(organism.dna): {'score': score}
            for organism, score in zip(self.model_tuner_max.population, range(self.model_tuner_min.population_size))
        }

        # Score the fitness
        self.model_tuner_max.score_fitness(unique_organisms)
        fitness_scores = [model.fitness for model in self.model_tuner_max.population]

        # Check that fitness scores are correctly assigned (highest fitness for the highest score)
        self.assertEqual(max(fitness_scores), 1.0, "Best fitness score should be 1.0 for the highest score.")
        self.assertEqual(min(fitness_scores), 0.0, "Worst fitness score should be 0.0 for the lowest score.")

        # TODO test that highest score has best fitness and vice versa


    def test_correct_best_and_worst_dna_identification_min(self):
        """Test that the best and worst DNA are correctly identified for minimization"""
        unique_organisms = {
            dna2str(organism.dna): {'score': score}
            for organism, score in zip(self.model_tuner_min.population, range(self.model_tuner_min.population_size))
        }

        self.model_tuner_min.score_fitness(unique_organisms)
        best_dna = self.model_tuner_min.metrics[-1]['best_dna']
        worst_dna = self.model_tuner_min.metrics[-1]['worst_dna']

        # Find expected best and worst DNA
        expected_best_dna = dna2str(self.model_tuner_min.population[0].dna)
        expected_worst_dna = dna2str(self.model_tuner_min.population[-1].dna)

        self.assertEqual(best_dna, expected_best_dna, "Best DNA should be the one with the lowest score for minimization.")
        self.assertEqual(worst_dna, expected_worst_dna, "Worst DNA should be the one with the highest score for minimization.")

    def test_correct_best_and_worst_dna_identification_max(self):
        """Test that the best and worst DNA are correctly identified for maximization"""
        unique_organisms = {
            dna2str(organism.dna): {'score': score}
            for organism, score in zip(self.model_tuner_max.population, range(self.model_tuner_min.population_size))
        }

        self.model_tuner_max.score_fitness(unique_organisms)
        best_dna = self.model_tuner_max.metrics[-1]['best_dna']
        worst_dna = self.model_tuner_max.metrics[-1]['worst_dna']

        # Find expected best and worst DNA
        expected_best_dna = dna2str(self.model_tuner_max.population[-1].dna)
        expected_worst_dna = dna2str(self.model_tuner_max.population[0].dna)

        self.assertEqual(best_dna, expected_best_dna, "Best DNA should be the one with the highest score for maximization.")
        self.assertEqual(worst_dna, expected_worst_dna, "Worst DNA should be the one with the lowest score for maximization.")


class TestModelTunerSelectionAndReproduction(unittest.TestCase):

    def setUp(self):
        # Sample data and model space setup with variation for testing
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
            'label': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Model space with variation in functions and arguments
        self.model_space = [
            [
                {'name': 'gene1', 'train': {'func': lambda x: x, 'inputs': 'x_train', 'outputs': 'output1', 'args': [ContinuousRange(0, 10)], 'kwargs': {'param1': ContinuousRange(0, 10)}}},
                {'name': 'gene2', 'train': {'func': lambda x: x**2, 'inputs': 'x_train', 'outputs': 'output2', 'args': [ContinuousRange(5, 15)], 'kwargs': {'param2': ContinuousRange(0, 10)}}}
            ],
            [
                {'name': 'gene3', 'train': {'func': lambda x: x + 1, 'inputs': 'output1', 'outputs': 'output3', 'args': [ContinuousRange(0, 10)], 'kwargs': {'param3': ContinuousRange(0, 10)}}},
                {'name': 'gene4', 'train': {'func': lambda x: x - 1, 'inputs': 'output2', 'outputs': 'output4', 'args': [ContinuousRange(5, 15)], 'kwargs': {'param4': ContinuousRange(0, 10)}}}
            ]
        ]
        self.model_tuner = ModelTuner(self.model_space, data, y_col='label', pop_size=1000)
        self.model_tuner.populate_init()

        # assign scores for testing elitism
        for i, organism in enumerate(self.model_tuner.population):
            organism.fitness = 1.0 - (i * 0.1)  # Higher fitness for earlier organisms

    def test_mutation_effects_with_probabilities(self):
        """Test that mutations occur at a rate consistent with given probabilities
        
        Statistical calculations get hairy when there are differing numbers of nucleotides in different gene options for the same gene slot.
        Therefore, we give all gene options in 1 slot the same number of nucleotides.

        Also if a gene did mutate, we have no visibility into the old gene, so we can't know nucleotide count unless we track parent genes.
        """
        # Set mutation probabilities
        gene_mutate_prob = 0.2
        nuc_mutate_prob = 0.5

        # Set elitism to 0 to isolate testing probabilities
        elitism = 0

        # Capture the original DNA of the entire population
        original_dna_set = {dna2str(organism.dna) for organism in self.model_tuner.population}
        
        # Perform reproduction with set mutation probabilities
        self.model_tuner.select_and_reproduce(
            elitism=elitism,
            gene_mutate_prob=gene_mutate_prob,
            nuc_mutate_prob=nuc_mutate_prob
        )

        # Capture the new DNA
        new_dna_list = [dna2str(organism.dna) for organism in self.model_tuner.population]

        # Count how many organisms have DNA that matches any in the original DNA set (i.e., unchanged)
        num_unchanged = sum(1 for dna in new_dna_list if dna in original_dna_set)

        # Calculate the probability that an organism remains unchanged
        # The probability that a gene does not mutate is (1 - gene_mutate_prob)
        # The probability that each nucleotide does not mutate is (1 - nuc_mutate_prob)
        n_mutatable_genes = len([gene for gene in self.model_space if len(gene) > 1])
        prob_no_gene_mutation = (1 - gene_mutate_prob) ** n_mutatable_genes

        n_mutatable_nucleotides = 0
        # IMPORTANT, we assume that all options for all genes in one slot of the model space have the same # of nucleotides.
        # See docstring, makes calculation easier.
        for gene_space in self.model_space:
            n_mutatable_nucleotides += len(gene_space[0]['train']['args'])
            n_mutatable_nucleotides += len(gene_space[0]['train']['kwargs'])

        prob_no_nucleotide_mutation = (1 - nuc_mutate_prob) ** n_mutatable_nucleotides
        prob_no_mutation = prob_no_gene_mutation * prob_no_nucleotide_mutation

        # Expected number of unchanged organisms
        population_size = len(self.model_tuner.population)
        expected_unchanged = population_size * prob_no_mutation
        std_dev = np.sqrt(population_size * prob_no_mutation * (1 - prob_no_mutation))  # Binomial

        # Use a confidence interval to check if the observed number of unchanged organisms is reasonable
        lower_bound = expected_unchanged - 2 * std_dev
        upper_bound = expected_unchanged + 2 * std_dev

        self.assertGreaterEqual(num_unchanged, lower_bound, "Observed number of unchanged organisms is lower than expected.")
        self.assertLessEqual(num_unchanged, upper_bound, "Observed number of unchanged organisms is higher than expected.")

    def test_elitism_with_forced_mutation(self):
        """Test that the top elite organisms' DNA is still present after reproduction with 100% mutation chance"""
        # Number of elite organisms to preserve
        elitism = 3

        # Capture the DNA of the top elite organisms
        original_top_elite_dna = {dna2str(organism.dna) for organism in sorted(self.model_tuner.population, reverse=True)[:elitism]}

        # Perform reproduction with 100% mutation chance
        self.model_tuner.select_and_reproduce(
            elitism=elitism,
            gene_mutate_prob=1.0,  # 100% chance of mutating the gene's function
            nuc_mutate_prob=1.0,   # 100% chance of mutating each gene's nucleotide
            max_discrete_shift=2,
            max_continuous_shift=0.05
        )

        # Collect the DNA of the new population
        new_population_dna = {dna2str(organism.dna) for organism in self.model_tuner.population}

        # Check that the DNA of the top elite organisms is still present in the new population
        self.assertTrue(original_top_elite_dna.issubset(new_population_dna), "The DNA of the top elite organisms should still be present in the new population.")


if __name__ == '__main__':
    unittest.main()
