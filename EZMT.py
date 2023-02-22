from functools import partial
import pandas as pd
import numpy as np
from random import choice, uniform, random, randint
import multiprocessing as mp
from sklearn.metrics import mean_squared_error as mse


class ModelTuner:

    def __init__(self, pool, data, y_col, generations=1, pop_size=20, method=None):
        # Generations can be used for batches of data and not for evolution
        if pool is None:
            self.pool = mp.Pool(8)
        else:
            self.pool = pool

        self.X_train, self.X_test, self.Y_train, self.Y_test = tts(data, y_col, 0.75)
        self.steps = []
        self.generations = generations
        self.population = [Model([], []) for i in range(pop_size)]
        self.results = dict()

    def add_step(self, step):
        """
        Add a step to the skeleton
        :param step:
        {
            name: some_name0
            func: some_func0,
            args: [
                [arg0_0_0, arg0_0_1],     <- Possible choices for first argument of first step in model
                [arg0_1_0, arg0_1_1]      <- Possible choices for second argument of first step in model
            ]
        }
        """

        step['func'] = partial(step_await, step['func'])
        self.steps.append(step_await)

    def populate_init(self, generation):
        """
        Generates a population by iterating through genes slots and randomly choosing pop_size# genes with replacement.
        In other words, generate each population member's first gene, then do everyone's second, etc
        """
        # Generate initial population
        if generation == 0:
            for step in self.steps:
                # One gene corresponds to one function in the chain of steps
                for i in range(len(self.population)):
                    gene = []
                    for nuc_options in step['args']:
                        # One nucleotide corresponds to one argument.
                        # One of multiple options (e.g. A, C, G, T) is chosen
                        gene.append(choice(nuc_options))
                    self.population[i].add_gene(gene)

    def populate_next(
            self,
            elitism=0,
            nuc_change_chance=0.1,
            reproduction='asexual'
    ):
        # Generate population from previous generation
        # keep top models
        self.population = sorted(self.population, reverse=True)
        new_pop = self.population[0:elitism]

        survivors = natural_selection(self.population)

        for _ in range(len(self.population) - elitism):
            if reproduction == 'asexual':
                child = choice(self.population)
            elif reproduction == 'sexual':
                parent1 = choice(self.population)
                parent2 = choice(self.population)
                child = parent1.mate(parent2)
            child = mutate(child, self.steps, nuc_change_chance, 1)
            new_pop.append(child)
        self.population = new_pop

    def train_population(self, data):
        self.results = {[]: data}
        for i, train_step in enumerate(self.steps):
            for model in self.population:
                # check if result of model with identical development up until this stage has already been calculated
                if model.dna in self.results.keys():
                    continue
                # get data with matching genes from previous stage of development and apply function + args of next gene
                args = model.dna[i]
                data = self.results[model.dna[:i]]
                self.results[model.dna] = self.pool.apply_async(train_step, args=(data, *args))

    def calc_fitness(self, criterion=mse):
        # I keep this if statement to remind me to add more criterion
        for model in self.population:
            model.fitness = criterion(self.data[self.y_col], self.results[model.dna])


    def run(self, criterion):
        for gen in range(self.generations):
            if gen == 0:
                self.populate_init(gen)
            else:
                self.populate_next()
            self.train_population(self.X_train)
            self.calc_fitness(criterion)

        if criterion == 'MSE':
            return min(self.population)


class Model:

    def __init__(self, dna, steps):
        self.dna = dna
        self.steps = steps
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def add_gene(self, gene):
        self.dna.append(gene)

    def dna2str(self):
        dna_str = list(map('|'.join, self.dna))
        dna_str = '//'.join(dna_str)
        return dna_str

    def mate(self, other):
        pass


# Assume if not pd.DataFrame, then it's a pool.apply_async still running
def step_await(func, data, *args):
    if not isinstance(data, pd.DataFrame):
        data = data.get()
    return func(data, *args)


def natural_selection(
        population,
        selection_method='weighted_prob',
        survival_variation=0.1,
):
    sv = survival_variation
    if selection_method == 'weighted_prob':
        return [
            m for m in population if m.fitness * (1 + uniform(-sv, sv)) > uniform(0, max(population).fitness)
        ]

def mutate(model, framework, prob, max_mag):
    # framework provides possible mutations of model
    # Until nucleotides can represent continuous variables, max_mag is positions from the current nucleotide's value
    for i in range(len(framework)):
        for j, nucleotide in enumerate(model.dna[i]):
            if random() < prob:
                index = framework[i]['args'][j].index(nucleotide) + randint(-max_mag, max_mag)
                index = min(0, max(len(framework[i]['args'][j]) - 1), index)
                model.dna[i][j] = framework[i]['args'][index]

if __name__ == '__main__':
    pass

"""
step 1. Build skeleton of model with the following structure:
    [
        {                                   <- First step starts
            name: some_name0
            func: some_func0,
            args: [
                [arg0_0_0, arg0_0_1],     <- Possible choices for first argument of first step in model
                [arg0_1_0, arg0_1_1]      <- Possible choices for second argument of first step in model
            ]
        }, 
        {                                   <- Second step starts
            name: some_name1
            func: some_func1
            args: [
                [arg1_0_0, arg1_0_1],     <- Possible choices for first argument of second step in model
                [arg1_1_0, arg1_1_1]      <- Possible choices for second argument of second step in model
            ]
        }
    ]
    Assume that some_func's has an additional argument before the args listed that will hold the data from the
        resulting step.
    If you need to choose from multiple functions, make a wrapper function that chooses one based on an argument.
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
    let functions change
    let there be nucleotides that represent continuous variables
    let mutation magnitude adjust during the run
"""
