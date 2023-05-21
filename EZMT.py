import numpy as np
import random
import multiprocessing as mp
from sklearn.model_selection import StratifiedKFold
import pprint
import time

pp = pprint.PrettyPrinter(indent=4)


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

    def populate_init(self):
        """
        Generates a population by iterating through gene slots and randomly choosing genes.
        In other words, generate each population member's first gene, then do everyone's second, etc
        """
        # Generate initial population
        pp.pprint(self.decision_points)
        for dp in self.decision_points:
            # One gene in the DNA corresponds to one decision function in the chain of decisions
            for i in range(len(self.population)):
                gene = []
                for nuc_options in dp['args']:
                    # One nucleotide corresponds to one argument.
                    # One of multiple options (e.g. A, C, G, T) is chosen
                    if isinstance(nuc_options, list):  # discrete options
                        gene.append(random.choice(nuc_options))
                    if isinstance(nuc_options, tuple):  # continuous options
                        gene.append(round(random.uniform(nuc_options[0], nuc_options[1]), 5))
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
                child = random.choice(survivors)
            elif reproduction == 'sexual':
                parent1, parent2 = random.sample(survivors, 2)
                child = parent1.mate(parent2)
            child = mutate(child, self.decision_points, nuc_change_chance, 1, 0.5)
            new_pop.append(child)
        self.population = new_pop

    def experience_population(self, state):
        # Just as we experience the universe, the universe experiences us
        # for each decision point, process only unique chains of decisions + args from first decision point to current

        unique_organisms = {'': state}

        for i, train_step in enumerate(self.decision_points):
            print('Processing all unique combinations of options for decision: ' + train_step['name'])

            counter = 0
            prev_unique = unique_organisms
            unique_organisms = dict()
            # Start all processes for this section of DNA. Only process unique decisions based on populations' DNAs
            for model in self.population:
                current_dna = dna2str(model.dna[:i+1])

                # check if identical series of decisions up until this stage has already started calculating
                if current_dna in unique_organisms.keys():
                    continue

                counter += 1

                prev_dna = dna2str(model.dna[:i])
                # state is a dict of that hold all the saved outputs from previous steps for later use
                try:
                    state = dict(prev_unique[prev_dna])  # Must wrap in dict to make a copy
                except KeyError:
                    quit()
                unique_organisms[current_dna] = self.pool.apply_async(
                    self.make_decision,
                    args=(i, model, state, train_step)
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
    def make_decision(i, organism, state, train_step):
        if isinstance(train_step['func'], str):
            f_parts = train_step['func'].split('.')
            func = state[f_parts[0]]
            for part in f_parts[1:]:
                if hasattr(func, part):
                    func = getattr(func, part)
                else:
                    raise Exception('Could not get ' + part + ' from ' + train_step['func'])
        else:
            func = train_step['func']
        # get data with matching genes from previous stage of development and apply function + args of next gene
        args = (*[state[inp] for inp in train_step['inputs']], *organism.dna[i])
        output = func(*args)

        # Update State
        ons = train_step['outputs']  # output names
        if ons:
            if len(ons) > 1:
                output = {o: output[j] for j, o in enumerate(ons)}
            elif len(ons) == 1:
                output = {o: output for o in ons}
            state = {**state, **output}

        return state

    def score_fitness(self, unique_organisms):
        # TODO fix nan. See if len(scores) == 1
        scores = [uo['score'] for uo in unique_organisms.values()]
        if len(scores) == 1:
            for model in self.population:
                dna = dna2str(model.dna)
                model.score = unique_organisms[dna]['score']
                model.fitness = 1
            return

        if self.goal == 'min':
            worst = max(scores)
            best = min(scores)
        else:
            best = max(scores)
            worst = min(scores)

        for model in self.population:
            dna = dna2str(model.dna)
            model.score = unique_organisms[dna]['score']
            model.fitness = (model.score - worst) / (best - worst)

    def run(self):
        for gen in range(self.generations):
            if gen == 0:
                self.populate_init()
            else:
                self.select_and_reproduce(
                    elitism=1,
                    nuc_change_chance=0.75
                )
            print('--------------------------------')
            print(f'Starting generation {gen}: ')
            t = time.time()
            # Get next fold of data for next generation
            x_train, x_test, y_train, y_test = next(self.data_fold_generator)
            results = self.experience_population(
                {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
            )
            
            self.score_fitness(results)


            print('Results of generation ' + str(gen))
            print('Run Time: ' + str(time.time() - t))
            print(f'Average Score: {sum([p.score for p in self.population]) / len(self.population)}')
            print(f'Unique members: ' + str(len(results.keys())))
            print('--------------------------------')
        # score is converted into fitness, which always follows highest-is-best
        return max(self.population)


class Organism:

    def __init__(self, dna=None):
        self.dna = dna
        self.score = 0
        self.fitness = 0
        self.knowledge = None

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return dna2str(self.dna) == dna2str(other.dna)

    def __hash__(self):
        return hash(dna2str(self.dna))

    def __str__(self):
        return dna2str(self.dna)

    def __repr__(self):
        return self.__str__()

    def add_gene(self, gene):
        self.dna.append(gene)

    def mate(self, other):
        pass


def dna2str(dna):
    dna_str = ''
    for gene in dna:
        dna_str += '/ '
        for i, nucleotide in enumerate(gene):
            sep = ' '
            if i < len(gene) - 1:
                sep = ',' + sep
            dna_str += str(nucleotide) + sep
    return dna_str


def natural_selection(
        population,
        survival_variation=0.1,
        survival_percentage=0.7,
        min_probability=1e-6
):
    sv = survival_variation
    prob_dist = np.array([p.fitness * (1 + random.uniform(-sv, sv)) for p in population])
    prob_dist = np.clip(prob_dist, min_probability, None)
    prob_dist /= sum(prob_dist)
    n_survivors = int(survival_percentage * len(population))

    return np.random.choice(population, size=n_survivors, replace=False, p=prob_dist)


def mutate(organism, framework, prob, max_disc_shift, max_cont_shift):
    """
    mutates the genes of an organism, making it make different decisions
    :param organism: object of class Organism
    :param framework: model of all possible decisions for all decision points
    :param prob: probability of mutation for each nucleotide
    :param max_disc_shift: max discrete shift = the max change in index of the nucleotide option if options are discrete
    :param max_cont_shift: max continuous shift = the max change in the value of a nucleotide if options are continuous
    max_cont_shift is a percentage of the range from min to max value of nucleotide
    :return: mutated organism
    """
    for i in range(len(framework)):
        for j, nucleotide in enumerate(organism.dna[i]):
            if random.random() > prob:
                continue
            nf = framework[i]['args'][j]  # framework for specific nucleotide
            if isinstance(nf, list):
                index = nf.index(nucleotide) + random.randint(-max_disc_shift, max_disc_shift)
                index = max(0, min(len(nf) - 1, index))
                organism.dna[i][j] = nf[index]
            elif isinstance(nf, tuple):
                nucleotide += (nf[1] - nf[0]) * random.uniform(-max_cont_shift, max_cont_shift)
                nucleotide = max(nf[0], min(nf[1], nucleotide))
                nucleotide = round(nucleotide, 5)
                organism.dna[i][j] = nucleotide
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
