from functools import partial
import pandas as pd
from random import choice, uniform, random, randint
import multiprocessing as mp
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import StratifiedShuffleSplit


class ModelTuner:

    def __init__(self, pool, data, y_col, generations=1, pop_size=20, goal='min'):
        # Generations can be used for batches of data and not for evolution
        if pool is None:
            self.pool = mp.Pool(8)
        else:
            self.pool = pool

        # Todo actually split this better. Stratified with samples = generations?
        train, test = data, data
        self.X_train, self.y_train = train.drop(columns=y_col), train[y_col]
        self.X_test, self.y_test = test.drop(columns=y_col), test[y_col]


        self.steps = []
        self.generations = generations
        self.population = [Model([]) for i in range(pop_size)]
        self.goal = goal
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

        #step['func'] = partial(step_await, step['func'])
        self.steps.append(step)

    def populate_init(self, generation):
        """
        Generates a population by iterating through genes slots and randomly choosing pop_size# genes with replacement.
        In other words, generate each population member's first gene, then do everyone's second, etc
        """
        # Generate initial population
        print(self.steps)
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
        survivors = list(set([*survivors, *new_pop]))

        for _ in range(len(self.population) - elitism):
            if reproduction == 'asexual':
                child = choice(survivors)
            elif reproduction == 'sexual':
                parent1 = choice(survivors)
                parent2 = choice(survivors)
                child = parent1.mate(parent2)
            child = mutate(child, self.steps, nuc_change_chance, 1)
            new_pop.append(child)
        self.population = new_pop

    def learn(self, data):
        self.results = {-1: {'': data}}
        for i, train_step in enumerate(self.steps):
            self.results[i] = dict()
            # Start all processes for this section of dna on all models
            for model in self.population:
                prev_dna = dna2str(model.dna[:i])
                current_dna = dna2str(model.dna[:i+1])

                # check if result of model with identical development up until this stage has already been calculated
                if current_dna in self.results[i].keys():
                    continue

                # get data with matching genes from previous stage of development and apply function + args of next gene
                args = model.dna[i]
                data = self.results[i-1][prev_dna]
                self.results[i][current_dna] = self.pool.apply_async(train_step['func'], args=(data, *args))

            # Wait for all processes for this section of dna to complete
            for dna, data in self.results[i].items():
                if isinstance(data, mp.pool.ApplyResult):
                    self.results[i][dna] = data.get()

    def score_fitness(self):
        n_genes = len(self.steps)
        max_score = max(self.results[n_genes-1].values())
        min_score = min(self.results[n_genes-1].values())
        if self.goal == 'min':
            # convert from lowest-is-best to highest-is-best
            def conv(score): return - (score - max_score) / (max_score - min_score)
        else:
            def conv(score): return (score - min_score) / (max_score - min_score)

        for model in self.population:
            dna = dna2str(model.dna)
            model.score = self.results[n_genes-1][dna]
            model.fitness = conv(self.results[n_genes-1][dna])

    def run(self):
        for gen in range(self.generations):
            if gen == 0:
                self.populate_init(gen)
            else:
                self.populate_next()
            self.learn(self.X_train)
            self.score_fitness()

        # score is converted into fitness, which always follows highest-is-best
        return max(self.population)


class Model:

    def __init__(self, dna=None):
        self.dna = dna
        self.score = 0
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return dna2str(self.dna) == dna2str(other.dna)

    def __hash__(self):
        return hash(dna2str(self.dna))

    def __str__(self):
        return dna2str(self.dna)

    def add_gene(self, gene):
        self.dna.append(gene)

    def mate(self, other):
        pass


def dna2str(dna):
    dna_str = ''
    for gene in dna:
        for nucleotide in gene:
            if isinstance(nucleotide, pd.Series):
                dna_str += nucleotide.name + '|'
            else:
                dna_str += str(nucleotide) + '|'
        dna_str += '//'
    return dna_str


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
        # Todo just do a choice function with weights and no replacement
        max_fitness = max([p.fitness for p in population])
        return [
            m for m in population if m.fitness * (1 + uniform(-sv, sv)) > uniform(0, max_fitness)
        ]


def mutate(model, framework, prob, max_mag):
    # framework provides possible mutations of model
    # Until nucleotides can represent continuous variables, max_mag is positions from the current nucleotide's value
    for i in range(len(framework)):
        for j, nucleotide in enumerate(model.dna[i]):
            if random() < prob:
                index = framework[i]['args'][j].index(nucleotide) + randint(-max_mag, max_mag)
                index = min(0, max(len(framework[i]['args'][j]) - 1, index))
                model.dna[i][j] = framework[i]['args'][j][index]
    return model


# below functions are for testing purposes


def do_nothing(data, n):
    return data


def add_cols(data, useless_variable):
    return data.sum(axis=1)


if __name__ == '__main__':
    df = pd.DataFrame({'a': [0, 0, 0, 0, 1, 1, 1, 1],
                       'b': [0, 0, 0, 0, 0, 0, 1, 1],
                       'y': [0, 0, 0, 0, 1, 1, 2, 2]})

    mt = ModelTuner(None, df, 'y')
    mt.add_step(
        {
            'name': 'do_nothing',
            'func': do_nothing,
            'args': [
                [0, 1, 2, 3, 4, 5]
            ]
        }
    )

    mt.add_step(
        {
            'name': 'add_cols',
            'func': add_cols,
            'args': [
                ['ooga', 'booga', 'elephant', 'preposterous', 'waka waka', 'bbb']
            ]
        }
    )

    mt.add_step(
        {
            'name': 'MSE',
            'func': mse,
            'args': [
                [df['y']]
            ]
        }
    )

    result = mt.run()
    print(dna2str(result.dna), result.score, result.fitness)


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
