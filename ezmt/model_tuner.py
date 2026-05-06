import asyncio
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pprint
import time
from copy import deepcopy

from ezmt.organism import Organism, dna2str
from ezmt.common_funcs import resolve_log_states
from ezmt.config_validation import validate_config
from ezmt.the_pickler import check_state_picklability

pp = pprint.PrettyPrinter(indent=4)


class ModelTuner:

    def __init__(
            self,
            model_space: list,
            hyperparam_space: dict,
            save_load_funcs: dict = None,
            data: pd.DataFrame = None,
            y_col: str = None,
            generations: int = 1,
            pop_size: int = 20,
            goal: str = 'min'
    ):
        # Generations can be used for batches of data and not for evolution
        self.model_space = validate_config(model_space, hyperparam_space)
        self.hyperparam_space = hyperparam_space
        self.save_load_funcs = save_load_funcs if save_load_funcs else {}
        self.gpu_semaphore = asyncio.Semaphore(1)  # Used to ensure only 1 process is accessing the GPU at a time
        self.data_fold_generator = generate_stratified_folds(data, y_col)
        self.generations = generations
        self.population_size = pop_size
        self.population = []
        self.goal = goal
        self.metrics = []

    def populate_init(self, run_name):
        # Generate initial population
        for _ in range(self.population_size):
            dna = choose_dna(self.model_space)
            hyperparams = choose_hyperparams(self.hyperparam_space)
            organism = Organism(run_name, dna, hyperparams, save_load_funcs=self.save_load_funcs)
            self.population.append(organism)

    def select_and_reproduce(
            self,
            elitism=1,
            reproduction='asexual',
            gene_mutate_prob=0.05,
            nuc_mutate_prob=0.1,
    ):
        # Generate population from previous generation
        # keep top models
        self.population = sorted(self.population, reverse=True)
        new_pop = self.population[0:elitism]
        for member in new_pop:
            member.reset()  # Apply the veil

        # Determine who survives and can reproduce
        survivors = natural_selection(self.population)
        # Reproduce the population
        for _ in range(len(self.population) - len(new_pop)):
            if reproduction == 'asexual':
                parent = random.choice(survivors)
                child = parent.reproduce()
            elif reproduction == 'sexual':
                parent1, parent2 = random.sample(survivors, 2)
                child = parent1.mate(parent2)
            else:
                raise ValueError(f'Cannot reproduce with reproduction type: {reproduction}')
            mutate(
                child,
                self.model_space,
                self.hyperparam_space,
                gene_mutate_prob,
                nuc_mutate_prob,
            )
            new_pop.append(child)
        self.population = new_pop

    async def experience_population(self, state, pool, log_states=False):
        # Just as we experience the universe, the universe experiences us
        # for each decision point, process only unique chains of decisions + args from first decision point to current
        states_to_log = resolve_log_states(log_states)

        unique_organisms = {'': state}

        for i in range(len(self.model_space)):
            print(f'Processing gene {i + 1}/{len(self.model_space)} for each organism')
            prev_unique = unique_organisms
            unique_organisms = dict()

            # Start all processes for this section of DNA. Only process unique decisions based on populations' DNAs
            for organism in sorted(
                    self.population,
                    key=lambda org: (False if org.dna[i]['train'] is None else True) and org.dna[i]['train']['gpu']
            ):
                current_dna = dna2str(organism.dna[:i + 1])
                print(f'Processing organism branch {current_dna}')
                # check if identical series of decisions up to this stage has already started calculating
                if current_dna in unique_organisms.keys():
                    continue

                prev_dna = dna2str(organism.dna[:i])
                # state is a dict of that hold all the saved outputs from previous steps for later use
                try:
                    # Must wrap in dict to make a shallow copy
                    # I don't think we need to make a deep copy because we only overwrite keys on the outer layer
                    state = dict(prev_unique[prev_dna])
                except KeyError:
                    raise KeyError(f'No state found for previous dna: {prev_dna}')

                is_gpu = organism.dna[i]['train']['gpu']
                if is_gpu:
                    async with self.gpu_semaphore:
                        new_state = asyncio.create_task(
                            organism.run_gene('train', i, state, pool, log_state=states_to_log is None or i in states_to_log)
                        )
                else:
                    new_state = asyncio.create_task(
                        organism.run_gene('train', i, state, pool, log_state=states_to_log is None or i in states_to_log)
                    )
                # TODO organisms with current step param run_in_parent_process=True should be sorted to the end of the organisms
                #  so that other async/parallel jobs can start first.
                #  Also it should be assigned False by default for train branch in config validation

                unique_organisms[current_dna] = new_state

            # Wait for all processes for this decision point to complete
            for dna, output in unique_organisms.items():
                unique_organisms[dna] = await output

        # Each organism "remembers" what it has processed
        # knowledge is saved when the organism is saved, and it is loaded later to use during inference
        for organism in self.population:
            organism.knowledge = unique_organisms[dna2str(organism.dna)]

        return unique_organisms


    def score_fitness(self, unique_organisms):
        # TODO fix nan
        scores = []
        if self.goal == 'min':
            worst = -np.inf
            worst_dna = None
            best = np.inf
            best_dna = None
            for dna, data in unique_organisms.items():
                scores.append(data['score'])
                if data['score'] < best:
                    best = data['score']
                    best_dna = dna
                if data['score'] > worst:
                    worst = data['score']
                    worst_dna = dna

        else:
            worst = np.inf
            worst_dna = None
            best = -np.inf
            best_dna = None
            for dna, data in unique_organisms.items():
                scores.append(data['score'])
                if data['score'] > best:
                    best = data['score']
                    best_dna = dna
                if data['score'] < worst:
                    worst = data['score']
                    worst_dna = dna

        self.metrics.append({'unique_organisms': len(unique_organisms.keys()),
                             'average': np.mean(scores),
                             'variance': np.var(scores),
                             'best': best,
                             'best_dna': best_dna,
                             'worst': worst,
                             'worst_dna': worst_dna})

        for model in self.population:
            dna = dna2str(model.dna)
            model.score = unique_organisms[dna]['score']
            scores.append(model.score)
            if len(unique_organisms) == 1:
                model.fitness = 1
            else:
                model.fitness = (model.score - worst) / (best - worst)

    async def run(self, run_name, log_states=False):
        pp.pprint(self.model_space)
        with ProcessPoolExecutor(8) as pool:
            for gen in range(self.generations):
                if gen == 0:
                    self.populate_init(run_name)
                else:
                    self.select_and_reproduce()
                print(f'Starting generation {gen + 1}/{self.generations}')
                print('POPULATION:')
                for organism in self.population:
                    print(organism)
                t = time.time()
                # Get next fold of data for next generation
                x_train, x_test, y_train, y_test = next(self.data_fold_generator)
                results = await self.experience_population(
                    {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test},
                    pool,
                    log_states=log_states
                )

                self.score_fitness(results)
                print('Run Time: ' + str(time.time() - t))
                pp.pprint(self.metrics[-1])
                print('--------------------------------')
                for model in self.population:
                    pp.pprint(model.dna)

            # score is converted into fitness, which always follows highest-is-best
            return max(self.population)


def natural_selection(
        population,
        survival_variation=0.1,
        survival_percentage=0.5,
        min_probability=0.01
):
    sv = survival_variation
    prob_dist = np.array([p.fitness * (1 + random.uniform(-sv, sv)) for p in population])
    prob_dist = np.clip(prob_dist, min_probability, None)
    prob_dist /= sum(prob_dist)
    n_survivors = int(survival_percentage * len(population))

    return np.random.choice(population, size=n_survivors, replace=False, p=prob_dist)


def mutate(organism, model_space, hyperparam_space, func_prob, nuc_prob):
    """
    mutates the genes of an organism, making it make different decisions

    :param organism: object of class Organism
    :param model_space:
    :param hyperparam_space: all possible values of parameters to plug into dna
    :param func_prob: probability of mutation of the gene's function
    :param nuc_prob: probability of mutation of each of gene's nucleotides
    :param max_disc_shift: max discrete shift = the max change in index of the nucleotide option if options are discrete
    :param max_cont_shift: max continuous shift = the max change in the value of a nucleotide if options are continuous
    max_cont_shift is a percentage of the range from min to max value of nucleotide
    """
    # modify organism.dna
    for i, gene in enumerate(organism.dna):
        gene_space = model_space[i]
        if len(gene_space) > 1 and random.random() <= func_prob:
            available_genes = [g for g in gene_space if g != organism.dna[i]]
            organism.dna[i] = choose_gene(available_genes)
            continue

    # modify organism.parameters
    for hp, val in organism.parameters.items():
        if random.random() <= nuc_prob:
            organism.parameters[hp] = hyperparam_space[hp].mutate(val)


def choose_dna(dna_space):
    dna = []
    for gene_space in dna_space:
        # choose a random function from supplied choices
        gene = choose_gene(gene_space)
        dna.append(gene)
    return dna


def choose_gene(gene_space):
    return deepcopy(random.choice(gene_space))


def choose_hyperparams(hyperparam_space):
    return {key: val.sample() for key, val in hyperparam_space.items()}


def generate_stratified_folds(data, y_col, n_splits=5):
    while data is None:
        yield None, None, None, None  # Assumed to be supplied within a step in the model space
    x = data.drop(y_col, axis=1)
    y = data[y_col]
    skf = StratifiedKFold(n_splits=n_splits)
    splits = list(skf.split(x, y))
    i = 0
    while True:
        train_idx, test_idx = splits[i % n_splits]
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield x_train, x_test, y_train, y_test
        i += 1


"""
Future TODO:
    Make ability to add steps before splitting data for ease of use with new data (Or user supplied generators?)
    let mutation magnitude & probability adjust during the run
"""
