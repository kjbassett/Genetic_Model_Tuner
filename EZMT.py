import numpy as np
import random
import multiprocessing as mp
from sklearn.model_selection import StratifiedKFold
import pprint
import time
from copy import deepcopy

from organism import Organism, dna2str

pp = pprint.PrettyPrinter(indent=4)


class ModelTuner:

    def __init__(self, model_space, data, y_col, generations=1, pop_size=20, goal='min'):
        # Generations can be used for batches of data and not for evolution
        self.model_space = model_space
        self.validate_input()
        self.pool = mp.Pool(8)
        self.data_fold_generator = generate_stratified_folds(data, generations, y_col)
        self.generations = generations
        self.population_size = pop_size
        self.population = []
        self.goal = goal
        self.metrics = []

    def validate_input(self):
        new_model_space = []
        names_seen = {}

        for gene_space in self.model_space:
            if isinstance(gene_space, dict):
                gene_space = [gene_space]  # Standardize so we can do random.choice

            new_gene_space = []
            for function_dict in gene_space:

                # function info in function dict
                if "train" not in function_dict and 'inference' not in function_dict:
                    self.validate_function(function_dict, names_seen)
                # function info different for training and inference. function infos in 'train' and 'inference' keys
                # validate train func
                elif 'train' in function_dict:
                    self.validate_function(function_dict['train'], names_seen)
                # validate inference func
                elif 'inference' in function_dict:
                    self.validate_function(function_dict['inference'], names_seen)
                # needs func, else needs train and/or inference
                else:
                    raise ValueError(f"Bad function dictionary config. No 'train', 'inference' or 'func' key found.")



                new_gene_space.append(function_dict)
            new_model_space.append(new_gene_space)

        self.model_space = new_model_space

    def validate_function(self, function_dict, names_seen):
        if not isinstance(function_dict, dict):
            raise TypeError(f"{function_dict} is not a dictionary.")
        if 'func' not in function_dict:
            raise ValueError(f"No 'func' key found in {function_dict}.")
        if not callable(function_dict['func']) and not isinstance(function_dict['func'], str):
            raise TypeError(f"'func' value in {function_dict} is not callable or string.")
        for io in ('inputs', 'outputs'):
            if io not in function_dict:
                function_dict[io] = []
                continue
            if isinstance(function_dict[io], str):
                function_dict[io] = [function_dict[io]]  # standardize
            if not hasattr(function_dict[io], '__iter__'):
                raise TypeError(f"'{io}' value in {function_dict} is not iterable.")
        if 'args' not in function_dict:
            function_dict['args'] = []
        else:
            if not isinstance(function_dict['args'], list):
                raise TypeError(f"'args' value in {function_dict} is not a list.")
            for arg in function_dict['args']:
                if not hasattr(arg, '__iter__'):
                    raise TypeError(f"Argument {arg} in 'args' value of {function_dict} is not iterable.")
        if 'kwargs' not in function_dict:
            function_dict['kwargs'] = dict()
        if not isinstance(function_dict['kwargs'], dict):
            raise TypeError(f"'kwargs' value in {function_dict} is not a dictionary.")
        for value in function_dict['kwargs'].values():
            if not hasattr(value, '__iter__'):
                raise TypeError(f"Value {value} in 'kwargs' value of {function_dict} is not iterable.")
        if 'name' not in function_dict:
            if callable(function_dict['func']):
                function_dict['name'] = function_dict['func'].__name__
            else:
                function_dict['name'] = function_dict['func']
        if function_dict['name'] in names_seen:
            names_seen[function_dict['name']] += 1
            function_dict['name'] += f'_{names_seen[function_dict["name"]]}'
        else:
            names_seen[function_dict['name']] = 0

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
            reproduction='asexual'
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
            mutate(child, self.model_space, 0.1, 1, 1, 0.05)
            new_pop.append(child)
        self.population = new_pop

    def experience_population(self, state):
        # Just as we experience the universe, the universe experiences us
        # for each decision point, process only unique chains of decisions + args from first decision point to current

        unique_organisms = {'': state}

        for i in range(len(self.model_space)):

            prev_unique = unique_organisms
            unique_organisms = dict()
            # Start all processes for this section of DNA. Only process unique decisions based on populations' DNAs
            for organism in self.population:
                current_dna = dna2str(organism.dna[:i + 1])

                # check if identical series of decisions up until this stage has already started calculating
                if current_dna in unique_organisms.keys():
                    continue

                prev_dna = dna2str(organism.dna[:i])
                # state is a dict of that hold all the saved outputs from previous steps for later use
                try:
                    state = dict(prev_unique[prev_dna])  # Must wrap in dict to make a copy
                except KeyError:
                    quit()
                unique_organisms[current_dna] = self.pool.apply_async(
                    self.make_decision,
                    args=(organism, i, state)
                )
            
            # Wait for all processes for this decision point to complete
            for dna, output in unique_organisms.items():
                if isinstance(output, mp.pool.ApplyResult):
                    unique_organisms[dna] = output.get()

        # Each organism "remembers" what it has processed
        for organism in self.population:
            organism.knowledge = unique_organisms[dna2str(organism.dna)]

        return unique_organisms

    @staticmethod
    def make_decision(organism, i, state):
        gene = organism.dna[i]
        func = organism.dna[i]['func']

        if isinstance(func, str):  # if str, get it from values of state ('model.run' => 'model' is a key in state)
            f = func.split('.')
            func = state[f[0]]
            for part in f[1:]:
                if hasattr(func, part):
                    func = getattr(func, part)
                else:
                    raise Exception('Could not get ' + part + ' from ' + organism.dna[i]['func'])

        # get data with matching genes from previous stage of development and apply function + args of next gene
        args = (*[state[inp] for inp in gene['inputs']], *gene['args'])
        output = func(*args, **gene['kwargs'])

        # Update State
        ons = gene['outputs']  # output names
        if ons:
            if len(ons) > 1:
                output = {o: output[j] for j, o in enumerate(ons)}
            elif len(ons) == 1:
                output = {o: output for o in ons}
            state = {**state, **output}

        return state

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

    def run(self):
        pp.pprint(self.model_space)
        for gen in range(self.generations):
            if gen == 0:
                self.populate_init()
            else:
                self.select_and_reproduce()
            print(f'Starting generation {gen+1}/{self.generations}')
            print('POPULATION:')
            for organism in self.population:
                print(organism)
            t = time.time()
            # Get next fold of data for next generation
            x_train, x_test, y_train, y_test = next(self.data_fold_generator)
            results = self.experience_population(
                {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
            )
            
            self.score_fitness(results)
            print('Run Time: ' + str(time.time() - t))
            pp.pprint(self.metrics[-1])
            print('--------------------------------')
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

        if len(gene_space) > 1 and random.uniform(0, 1) <= func_prob:
            organism.dna[i] = choose_gene_from_space(gene_space)
            # choosing a new gene (function) chooses random nucleotides (args), so no need to mutate this gene further
            continue

        # find corresponding gene_space in model_space
        gene_variant = next((space for space in gene_space if space['name'] == gene['name']), None)
        if gene_variant is None:
            raise ValueError(f'Could not find corresponding gene_space in model_space.\n {[dna2str(gene)]}')

        # Modify nucleotides slightly but still remain in same gene variant
        for j, nucleotide in enumerate(gene['args']):
            if random.random() > nuc_prob:
                continue
            nucleotide_space = gene_variant['args'][j]  # space for specific nucleotide
            if isinstance(nucleotide_space, tuple):
                gene['args'][j] += (nucleotide_space[1] - nucleotide_space[0]) * random.uniform(-max_cont_shift, max_cont_shift)
                gene['args'][j] = max(nucleotide_space[0], min(nucleotide_space[1], gene['args'][j]))
                gene['args'][j] = round(gene['args'][j], 5)
            else:
                index = nucleotide_space.index(nucleotide) + random.randint(-max_disc_shift, max_disc_shift)
                index = max(0, min(len(nucleotide_space) - 1, index))
                gene['args'][j] = nucleotide_space[index]


def choose_nucleotides(nucleotide_space):
    nucleotides = {}

    if 'args' in nucleotide_space:
        nucleotides['args'] = [round(random.uniform(*arg), 5) if isinstance(arg, tuple) else random.choice(arg)
                               for arg in nucleotide_space['args']]

    if 'kwargs' in nucleotide_space:
        nucleotides['kwargs'] = {
            key: round(random.uniform(*value), 5) if isinstance(value, tuple) else random.choice(value)
            for key, value in nucleotide_space['kwargs'].items()
        }

    return nucleotides


def choose_gene_from_space(gene_space):
    nucleotide_space = random.choice(gene_space)
    nucleotides = choose_nucleotides(nucleotide_space)
    # add chosen nucleotides to the gene
    gene = deepcopy(nucleotide_space)
    gene.update(nucleotides)
    return gene


def generate_stratified_folds(data, n_splits, y_col):
    x = data.drop(y_col, axis=1)
    y = data[y_col]
    skf = StratifiedKFold(n_splits=n_splits)
    for train_idx, test_idx in skf.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield x_train, x_test, y_train, y_test


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
            Repeat step 3 and 4 until stopping condition it met
            Base the parameters of the next generation off of the previous generation
                according to some function of their performance scores

Future TODO:
    Make ability to add steps before splitting data for ease of use with new data
    let functions change
    let there be nucleotides that represent continuous variables
    let mutation magnitude adjust during the run
"""
