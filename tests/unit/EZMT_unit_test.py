from copy import deepcopy
import unittest
from unittest.mock import patch, MagicMock
from ezmt.model_tuner import ModelTuner, dna2str
from ezmt.organism import Organism
from ezmt.hyperparameters import ContinuousRange
import pandas as pd
import numpy as np
import time


class TestModelTunerPopulationInitialization(unittest.TestCase):

    def setUp(self):
        # Sample data and model space setup for testing
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
            'label': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        dna_space = [
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
                           'args': ['example_hp'], 'kwargs': {}}}
            ]
        ]
        hyperparam_space = {'example_hp': ContinuousRange(0, 10)}
        self.model_tuner = ModelTuner(dna_space, hyperparam_space, data=data, y_col='label', pop_size=20)

    def test_population_size(self):
        """Test that population is initialized with the correct size"""
        self.model_tuner.populate_init('test')
        self.assertEqual(len(self.model_tuner.population), 20, "Population size does not match specified pop_size.")

    def test_gene_length(self):
        """Test that each organism in the population is initialized with the correct gene structure"""
        self.model_tuner.populate_init('test')
        for organism in self.model_tuner.population:
            self.assertIsInstance(organism, Organism, "Population should contain Organism instances.")
            self.assertEqual(len(organism.dna), len(self.model_tuner.model_space),
                             "Each organism's DNA length should match the model space length.")

    def test_gene_options(self):
        self.model_tuner.populate_init('test')
        for organism in self.model_tuner.population:
            for gene, gene_space in zip(organism.dna, self.model_tuner.model_space):
                self.assertIn(gene['name'], [gs['name'] for gs in gene_space],
                              "Organism gene names should match options in model space.")

    def test_dna_diversity(self):
        """Test that the initial population has varied genes based on model space options"""
        self.model_tuner.populate_init('test')
        # Collect DNA strings to check for diversity in initial population
        gene_pool = {str(organism.dna) for organism in self.model_tuner.population}
        # A population with high diversity should have multiple unique DNA sequences
        self.assertEqual(len(gene_pool), 2, "Population appears to lack diversity.")

    def test_gene_diversity(self):
        """Test that each gene + arg combination has a variety of choices in the population"""
        self.model_tuner.populate_init('test')
        # Collect DNA strings to check for diversity in initial population
        for i in range(len(self.model_tuner.model_space)):
            gene_pool = {str(organism.dna[i]) for organism in self.model_tuner.population}
            if i == 0:
                self.assertEqual(len(gene_pool), 2, "The total unique genes for index 0 should be 2.")
            elif i == 1:
                self.assertEqual(len(gene_pool), 1, "The total unique genes for index 0 should be 1.")
    def test_parameter_diversity(self):
        """Test that each gene + arg combination has a variety of choices in the population"""
        self.model_tuner.populate_init('test')
        # Collect DNA strings to check for diversity in initial population
        gene_pool = {str(organism.parameters) for organism in self.model_tuner.population}
        self.assertEqual(len(gene_pool), 20)

    def test_gene_initialization_with_random_choices(self):
        """Test that all gene names are represented in the initial population"""
        self.model_tuner.populate_init('test')
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
        self.model_tuner_min = ModelTuner(model_space, {}, data=data, y_col='label', goal='min')
        self.model_tuner_max = ModelTuner(model_space, {}, data=data, y_col='label', goal='max')

        # Populate the initial population for testing
        self.model_tuner_min.populate_init('test')
        self.model_tuner_max.populate_init('test')

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
                {'name': 'gene1', 'train': {'func': lambda x: x, 'inputs': 'x_train', 'outputs': 'output1', 'args': ['arg1'], 'kwargs': {'param1': 'kwarg1'}}},
                {'name': 'gene2', 'train': {'func': lambda x: x**2, 'inputs': 'x_train', 'outputs': 'output2', 'args': ['arg2'], 'kwargs': {'param2': 'kwarg2'}}}
            ],
            [
                {'name': 'gene3', 'train': {'func': lambda x: x + 1, 'inputs': 'output1', 'outputs': 'output3', 'args': ['arg3'], 'kwargs': {'param3': 'kwarg3'}}},
                {'name': 'gene4', 'train': {'func': lambda x: x - 1, 'inputs': 'output2', 'outputs': 'output4', 'args': ['arg4'], 'kwargs': {'param4': 'kwarg4'}}}
            ]
        ]
        self.hyperparam_space = {
            'arg1': ContinuousRange(0, 10),
            'arg2': ContinuousRange(5, 15),
            'arg3': ContinuousRange(0, 10),
            'arg4': ContinuousRange(5, 15),
            'kwarg1': ContinuousRange(0, 10),
            'kwarg2': ContinuousRange(0, 10),
            'kwarg3': ContinuousRange(0, 10),
            'kwarg4': ContinuousRange(0, 10)
        }
        self.model_tuner = ModelTuner(self.model_space, self.hyperparam_space, data=data, y_col='label', pop_size=1000)
        self.model_tuner.populate_init('test')

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
        nuc_mutate_prob = 0.1

        # Set elitism to 0 to isolate testing probabilities
        elitism = 0

        # Capture the original DNA of the entire population
        original_organisms = deepcopy(self.model_tuner.population)

        # Perform reproduction with set mutation probabilities
        self.model_tuner.select_and_reproduce(
            elitism=elitism,
            gene_mutate_prob=gene_mutate_prob,
            nuc_mutate_prob=nuc_mutate_prob
        )

        # Count how many organisms have DNA that matches any in the original DNA set (i.e., unchanged)
        num_unchanged = len([o for o in self.model_tuner.population if o in original_organisms])

        # Calculate the probability that an organism remains unchanged
        # The probability that a gene does not mutate is (1 - gene_mutate_prob)
        # The probability that each parameter does not mutate is (1 - nuc_mutate_prob)
        n_mutatable_genes = sum(1 for slot in self.model_space if len(slot) > 1)
        prob_no_gene_mutation = (1 - gene_mutate_prob) ** n_mutatable_genes
        print(prob_no_gene_mutation)

        n_mutatable_nucleotides = len(self.hyperparam_space)
        prob_no_nucleotide_mutation = (1 - nuc_mutate_prob) ** n_mutatable_nucleotides
        print(prob_no_nucleotide_mutation)

        prob_no_mutation = prob_no_gene_mutation * prob_no_nucleotide_mutation

        # Expected number of unchanged organisms
        population_size = len(self.model_tuner.population)
        print(population_size)

        # Use a confidence interval to check if the observed number of unchanged organisms is reasonable
        expected_unchanged = population_size * prob_no_mutation
        std_dev = np.sqrt(population_size * prob_no_mutation * (1 - prob_no_mutation))  # Binomial
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
            nuc_mutate_prob=1.0    # 100% chance of mutating each gene's nucleotide
        )

        # Collect the DNA of the new population
        new_population_dna = {dna2str(organism.dna) for organism in self.model_tuner.population}

        # Check that the DNA of the top elite organisms is still present in the new population
        self.assertTrue(original_top_elite_dna.issubset(new_population_dna), "The DNA of the top elite organisms should still be present in the new population.")


class TestModelTunerExperiencePopulation(unittest.IsolatedAsyncioTestCase):
    """GPU branches are dispatched last, so CPU work overlaps with them.

    A GPU gene holds the semaphore for its whole run. Starting one before the
    CPU branches have been dispatched leaves those branches waiting behind it
    for no reason, so the order the branches are handed to run_gene matters.

    This asserts the dispatch order rather than wall-clock time: the branches
    genuinely run concurrently once dispatched, so timing them is a race.
    """

    def setUp(self):
        self.model_space = [
            [
                {'name': 'gpu_gene', 'train': {'func': time.time, 'outputs': 'output1', 'gpu': True}},
                {'name': 'cpu_gene', 'train': {'func': time.time, 'outputs': 'output1'}},
            ],
            [
                {'name': 'gene3', 'train': {'func': time.time, 'outputs': 'output2', 'gpu': True}}
            ]
        ]
        self.model_tuner = ModelTuner(self.model_space, {}, pop_size=2)
        self.model_tuner.populate_init('test')
        # populate_init samples at random; the ordering only means something
        # with one of each kind present.
        for organism, gene_space in zip(self.model_tuner.population, self.model_space[0]):
            organism.dna[0] = deepcopy(gene_space)

    async def _dispatch_order(self):
        """Return the gene name of each branch in the order run_gene saw it."""
        order = []
        real_run_gene = Organism.run_gene

        async def spy(organism, mode, gene_index, state, pool=None, log_state=False):
            if gene_index == 0:
                order.append(organism.dna[0]['name'])
            return await real_run_gene(
                organism, mode, gene_index, state, pool, log_state
            )

        with patch.object(Organism, 'run_gene', spy):
            await self.model_tuner.experience_population({}, None)
        return order

    async def test_cpu_branches_are_dispatched_before_gpu_branches(self):
        # Act
        order = await self._dispatch_order()
        # Assert
        self.assertEqual(order, ['cpu_gene', 'gpu_gene'])

    async def test_the_order_does_not_depend_on_the_population_order(self):
        # Arrange - the population arrives in whatever order selection left it,
        # so the sort has to impose the order rather than preserve it.
        self.model_tuner.population.reverse()
        # Act
        order = await self._dispatch_order()
        # Assert
        self.assertEqual(order, ['cpu_gene', 'gpu_gene'])

    async def test_only_unique_branches_are_run(self):
        # Arrange - twenty organisms sharing two branches must not do twenty
        # branches' worth of work.
        self.model_tuner.population = [
            self.model_tuner.population[i % 2] for i in range(20)
        ]
        # Act
        order = await self._dispatch_order()
        # Assert
        self.assertEqual(len(order), 2)


if __name__ == '__main__':
    unittest.main()
