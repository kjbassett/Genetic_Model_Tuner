from functools import partial
import pandas as pd
from random import choice, uniform, random, randint
import multiprocessing as mp
from sklearn.model_selection import StratifiedKFold


class ModelTuner:

    def __init__(self, pool, data, y_col, generations=1, pop_size=20, goal='min'):
        # Generations can be used for batches of data and not for evolution
        if pool is None:
            self.pool = mp.Pool(8)
        else:
            self.pool = pool

        self.data_fold_generator = generate_stratified_folds(data, generations, y_col)

        self.decision_points = []
        self.generations = generations
        self.population = [Organism([]) for _ in range(pop_size)]
        self.goal = goal

    def add_decision_point(self, func, name=None, inputs=None, outputs=None, args=None):
        """
        Add step to skeleton (self.steps)
        :param func:
        :param inputs: names of the keys whose values must be retrieved from state.
                        Will be supplied to func in the order that they are listed
        :param outputs: names of the keys that will hold the outputs in state.
                        Must be the same length and order as outputs of func
        :param args: [
                [arg0_0_0, arg0_0_1],    <- Possible choices for first argument after input args of first step in model
                (arg0_1_0, arg0_1_1)     <- Range of choices for second argument after input args of first step in model
            ]
        :param name: name of the function, if None, func.__name__
        :return: None
        """

        if name is None:
            name = func if isinstance(func, str) else func.__name__
        if outputs is None:
            outputs = []
        elif isinstance(outputs, str):
            outputs = [outputs]
        if inputs is None:
            inputs = []
        elif isinstance(inputs, str):
            inputs = [inputs]
        if args is None:
            args = []

        decision_point = {
            'name': name,
            'func': func,
            'inputs': inputs,
            'args': args,
            'outputs': outputs
        }

        self.decision_points.append(decision_point)

    def populate_init(self, generation):
        """
        Generates a population by iterating through genes slots and randomly choosing genes.
        In other words, generate each population member's first gene, then do everyone's second, etc
        """
        # Generate initial population
        print(self.decision_points)
        if generation == 0:
            for step in self.decision_points:
                # One gene in the DNA corresponds to one decision function in the chain of decisions
                for i in range(len(self.population)):
                    gene = []
                    for nuc_options in step['args']:
                        # One nucleotide corresponds to one argument.
                        # One of multiple options (e.g. A, C, G, T) is chosen
                        gene.append(choice(nuc_options))
                    self.population[i].add_gene(gene)

    def select_and_reproduce(
            self,
            elitism=0,
            nuc_change_chance=0.1,
            reproduction='asexual'
    ):
        # Generate population from previous generation
        # keep top models
        self.population = sorted(self.population, reverse=True)
        new_pop = self.population[0:elitism]

        # Determine who survives
        survivors = natural_selection(self.population)
        survivors = list({*survivors, *new_pop})

        # Reproduce the population
        for _ in range(len(self.population) - elitism):
            if reproduction == 'asexual':
                child = choice(survivors)
            elif reproduction == 'sexual':
                parent1 = choice(survivors)
                parent2 = choice(survivors)
                child = parent1.mate(parent2)
            child = mutate(child, self.decision_points, nuc_change_chance, 1)
            new_pop.append(child)
        self.population = new_pop

    def experience_population(self, state):
        prev_combinations = {'': state}
        for i, train_step in enumerate(self.decision_points):
            print('Processing all unique combinations of options for decision: ' + train_step['name'])
            unique_combinations = dict()

            counter = 0
            # Start all processes for this section of dna. Only process unique decisions based on populations' dnas
            for m, model in enumerate(self.population):
                print(m)
                current_dna = dna2str(model.dna[:i+1])

                # check if identical series of decisions up until this stage has already started calculating
                if current_dna in unique_combinations.keys():
                    continue

                counter += 1

                prev_dna = dna2str(model.dna[:i])
                # state is a dict of that hold all the saved outputs from previous steps for later use
                state = dict(prev_combinations[prev_dna])  # Must wrap in dict to make a copy
                unique_combinations[current_dna] = self.pool.apply_async(self.make_decision, args=(i, model, state, train_step))

            print('Started ' + str(counter) + ' processes')
            
            # Wait for all processes from this section of dna to complete
            for dna, output in unique_combinations.items():
                if isinstance(output, mp.pool.ApplyResult):
                    unique_combinations[dna] = output.get()

            prev_combinations = unique_combinations

        return unique_combinations

    @staticmethod
    def make_decision(i, model, state, train_step):
        if isinstance(train_step['func'], str):
            f_parts = train_step['func'].split('.')
            func = state[f_parts[0]]
            for part in f_parts:
                if hasattr(func, part):
                    func = getattr(func, part)
                else:
                    raise Exception('Could not get ' + train_step['func'] + ' from state')
        else:
            func = train_step['func']
        # get data with matching genes from previous stage of development and apply function + args of next gene
        args = (*[state[inp] for inp in train_step['inputs']], *model.dna[i])
        output = func(*args)

        # if last step, just return the result (which should just be a score)
        if i >= len(model.dna) - 1:
            return output

        # Update State
        ons = train_step['outputs']  # output names
        if ons:
            if len(ons) > 1:
                output = {o: output[j] for j, o in enumerate(ons)}
            elif len(ons) == 1:
                output = {o: output for o in ons}
            state = {**state, **output}

        return state

    def score_fitness(self, results):
        if self.goal == 'min':
            worst = max(results.values())
            best = min(results.values())
        else:
            best = max(results.values())
            worst = min(results.values())

        for model in self.population:
            dna = dna2str(model.dna)
            model.score = results[dna]
            model.fitness = (results[dna] - worst) / (best - worst)

    def run(self):
        for gen in range(self.generations):
            if gen == 0:
                self.populate_init(gen)
            else:
                self.select_and_reproduce()

            # Get next fold of data for next generation
            x_train, x_test, y_train, y_test = next(self.data_fold_generator)
            results = self.experience_population(
                {'x_train': x_train, 'x_test': x_test, 'y_train': y_train}
            )
            
            self.score_fitness(results)

        # score is converted into fitness, which always follows highest-is-best
        return max(self.population)


class Organism:

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


def mutate(organism, framework, prob, max_mag):
    # framework = decision points. Remember a gene determines a decision
    # framework provide possible mutations of organism's genes
    # Until nucleotides can represent continuous variables, max_mag is positions from the current nucleotide's value
    for i in range(len(framework)):
        for j, nucleotide in enumerate(organism.dna[i]):
            if random() < prob:
                index = framework[i]['args'][j].index(nucleotide) + randint(-max_mag, max_mag)
                index = min(0, max(len(framework[i]['args'][j]) - 1, index))
                organism.dna[i][j] = framework[i]['args'][j][index]
    return organism


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
    Make ability to add steps before splitting data for ease of use with new data
    let functions change
    let there be nucleotides that represent continuous variables
    let mutation magnitude adjust during the run
"""
