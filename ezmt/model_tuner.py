import asyncio
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, Future
from sklearn.model_selection import StratifiedKFold
import pprint
import time
import inspect
from copy import deepcopy
import functools

from organism import Organism, dna2str
from config_validation import validate_config, ContinuousRange

pp = pprint.PrettyPrinter(indent=4)


class ModelTuner:

    def __init__(self, model_space, data, y_col, generations=1, pop_size=20, goal='min'):
        # Generations can be used for batches of data and not for evolution
        self.model_space = validate_config(model_space)
        self.gpu_semaphore = asyncio.Semaphore(1)
        self.data_fold_generator = generate_stratified_folds(data, y_col)
        self.generations = generations
        self.population_size = pop_size
        self.population = []
        self.goal = goal
        self.metrics = []

    def populate_init(self):
        # Generate initial population
        for _ in range(self.population_size):
            organism = Organism()
            for gene_space in self.model_space:
                gene = choose_gene_from_space(gene_space)  # Chooses nucleotide space and then specifies nucleotides
                organism.add_gene(gene)
            self.population.append(organism)

    def select_and_reproduce(
            self,
            elitism=1,
            reproduction='asexual',
            gene_mutate_prob=0.05,
            nuc_mutate_prob=0.1,
            max_discrete_shift=2,
            max_continuous_shift=0.05
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
            mutate(child, self.model_space, gene_mutate_prob, nuc_mutate_prob, max_discrete_shift, max_continuous_shift)
            new_pop.append(child)
        self.population = new_pop

    async def experience_population(self, state, pool):
        # Just as we experience the universe, the universe experiences us
        # for each decision point, process only unique chains of decisions + args from first decision point to current

        unique_organisms = {'': state}

        for i in range(len(self.model_space)):
            prev_unique = unique_organisms
            unique_organisms = dict()

            # Start all processes for this section of DNA. Only process unique decisions based on populations' DNAs
            for organism in sorted(
                    self.population,
                    key=lambda org: (False if org.dna[i]['train'] is None else True) and org.dna[i]['train']['gpu']
            ):
                current_dna = dna2str(organism.dna[:i + 1])

                # check if identical series of decisions up until this stage has already started calculating
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

                new_state = await self.run_gene(organism, i, state, pool)
                unique_organisms[current_dna] = new_state

            # Wait for all processes for this decision point to complete
            for dna, output in unique_organisms.items():
                if isinstance(output, Future):
                    unique_organisms[dna] = output.result()

        # Each organism "remembers" what it has processed
        # knowledge is saved when the organism is saved, and it is loaded later to use during inference
        for organism in self.population:
            organism.knowledge = unique_organisms[dna2str(organism.dna)]

        return unique_organisms

    async def run_gene(self, organism, gene_index, state, pool):
        if not organism.dna[gene_index]['train']:
            return state

        func = organism.dna[gene_index]['train']['func']
        is_async = inspect.iscoroutinefunction(func)
        is_gpu = organism.dna[gene_index]['train']['gpu']

        # If GPU is used, process serially to avoid excessive context switching with the GPU
        if is_gpu:
            async with self.gpu_semaphore:
                if is_async:
                    # Async GPU
                    return await organism.make_decision_async('train', gene_index, state)
                else:
                    # Sync GPU
                    return await asyncio.to_thread(organism.make_decision, 'train', gene_index, state)
        loop = asyncio.get_running_loop()
        if is_async:
            # Async CPU
            return await loop.run_in_executor(
                pool,
                functools.partial(run_async_in_process, organism.make_decision_async, 'train', gene_index, state)
            )
        else:
            # Sync CPU
            return await loop.run_in_executor(
                pool,
                organism.make_decision, 'train', gene_index, state
            )

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

    async def run(self):
        pp.pprint(self.model_space)
        with ProcessPoolExecutor(8) as pool:
            for gen in range(self.generations):
                if gen == 0:
                    self.populate_init()
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
                    pool
                )

                self.score_fitness(results)
                print('Run Time: ' + str(time.time() - t))
                pp.pprint(self.metrics[-1])
                print('--------------------------------')
                for model in self.population:
                    pp.pprint(model.dna)

            # score is converted into fitness, which always follows highest-is-best
            return max(self.population)


def run_async_in_process(async_func, *args, **kwargs):
    return asyncio.run(async_func(*args, **kwargs))


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


def mutate(organism, model_space, func_prob, nuc_prob, max_disc_shift, max_cont_shift):
    """
    mutates the genes of an organism, making it make different decisions

    :param model_space:
    :param organism: object of class Organism
    :param func_prob: probability of mutation of the gene's function
    :param nuc_prob: probability of mutation of each of gene's nucleotides
    :param max_disc_shift: max discrete shift = the max change in index of the nucleotide option if options are discrete
    :param max_cont_shift: max continuous shift = the max change in the value of a nucleotide if options are continuous
    max_cont_shift is a percentage of the range from min to max value of nucleotide
    """
    for i, gene in enumerate(organism.dna):
        gene_space = model_space[i]
        #
        #  you left off here because it needs to mutate train version

        if len(gene_space) > 1 and random.uniform(0, 1) <= func_prob:
            organism.dna[i] = choose_gene_from_space(gene_space)
            # choosing a new gene (function) chooses random nucleotides (args), so no need to mutate this gene further
            continue

        # find corresponding gene_space in model_space
        gene_variant = next((space for space in gene_space if space['name'] == gene['name']), None)
        if gene_variant is None:
            raise ValueError(f'Could not find corresponding gene_space in model_space.\n {[dna2str(gene)]}')

        gene = gene['train']
        gene_variant = gene_variant['train']
        if gene is None:
            continue

        # Modify nucleotides slightly but still remain in same gene variant
        for ak in ['args', 'kwargs']:
            for j, nucleotide in enumerate(gene[ak]) if ak == 'args' else gene[ak].items():
                if random.random() > nuc_prob:
                    continue
                nucleotide_space = gene_variant[ak][j]  # space for specific nucleotide
                if isinstance(nucleotide_space, ContinuousRange):
                    gene[ak][j] += (nucleotide_space.end - nucleotide_space.start) * random.uniform(-max_cont_shift,
                                                                                                    max_cont_shift)
                    gene[ak][j] = max(nucleotide_space.start, min(nucleotide_space.end, gene[ak][j]))
                    gene[ak][j] = round(gene[ak][j], 5)
                else:
                    index = nucleotide_space.index(nucleotide) + random.randint(-max_disc_shift, max_disc_shift)
                    index = max(0, min(len(nucleotide_space) - 1, index))
                    gene[ak][j] = nucleotide_space[index]


def choose_nucleotides(nucleotide_space):
    # Choose specific arg and kwarg values from arg and kwarg options
    nucleotides = {'args': [
        arg.sample() if isinstance(arg, ContinuousRange)
        else random.choice(arg)
        for arg in nucleotide_space['args']
    ], 'kwargs': {
        key: value.sample() if isinstance(value, ContinuousRange)
        else random.choice(value)
        for key, value in nucleotide_space['kwargs'].items()
    }}
    return nucleotides


def choose_gene_from_space(gene_space):
    # choose a random function from supplied choices
    nucleotide_space = random.choice(gene_space)
    gene = deepcopy(nucleotide_space)
    if gene["train"] is None:
        return gene
    nucleotides = choose_nucleotides(gene["train"])
    # add chosen nucleotides to the gene
    gene['train'].update(nucleotides)
    return gene


def generate_stratified_folds(data, y_col, n_splits=5):
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
step 1. Build skeleton of model with the following structure:
    [
        {                                   <- First step starts
            name: some_name0
            func: some_func0,
            args: [
                [arg0_0_0, arg0_0_1],     <- Possible choices for first argument of first step in model (list)
                (arg0_1_min, arg0_1_max)  <- Range of choices for second argument of first step in model (tuple)
            ]
        }, 
        {                                   <- Second step starts
            name: some_name1
            func: some_func1
            args: [
                (arg1_0_min, arg1_0_max), <- Range of choices for first argument of second step in model (tuple)
                [arg1_1_0, arg1_1_1]      <- Possible choices for second argument of second step in model (list)
            ]
        }
    ]
    Assume that some_func's has an additional argument before the args listed that will hold the data from the
        resulting step.
    Each member of the population can be identified by their DNA
    '0|0.6|False|a//1|True' => arg0_0|arg0_1|arg0_2|arg0_3//arg1_0|arg1_1
step 2. Choose some subset of all possible combinations of choices from skeleton in step 1.
step 3. Run:
    For each step
        For each *unique* combination of inputs and outputs according to subset from step 2
            calculate new data
    (In other words, don't do extra work! If 100 models have 2 choices for first step, do 2 calculations, not 100!)
step 4. Evaluate all models performance
step 5. If running multiple generations
            Repeat step 3 and 4 until stopping condition is met
            Base the parameters of the next generation off of the previous generation
                according to some function of their performance scores

Future TODO:
    Make ability to add steps before splitting data for ease of use with new data
    let mutation magnitude adjust during the run
"""
