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

        self.steps = []
        self.generations = generations
        self.population = [Model([]) for _ in range(pop_size)]
        self.goal = goal

    def add_step(self, func, inputs, outputs=None, *args, name=None):
        """
        Add step to skeleton (self.steps)
        :param func:
        :param inputs: names of the keys whose values must be retrieved from state.
                        Will be supplied to func in the order that they are listed
        :param outputs: names of the keys that will hold the outputs in state.
                        Must be the same length and order as outputs of func
        :param args: [
                [arg0_0_0, arg0_0_1],    <- Possible choices for first argument after input args of first step in model
                [arg0_1_0, arg0_1_1]     <- Possible choices for second argument after input args of first step in model
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

        step = {
            'name': name,
            'func': func,
            'inputs': inputs,
            'args': args,
            'outputs': outputs
        }

        self.steps.append(step)

    def populate_init(self, generation):
        """
        Generates a population by iterating through genes slots and randomly choosing genes.
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
        survivors = list({*survivors, *new_pop})

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

    def learn(self, state):
        prev_results = {'': state}
        for i, train_step in enumerate(self.steps):
            print('Experiencing ' + train_step['name'])
            results = dict()

            counter = 0
            # Start all processes for this section of dna on all models
            for m, model in enumerate(self.population):
                print(m)
                current_dna = dna2str(model.dna[:i+1])

                # check if result of model with identical development up until this stage has already been calculated
                if current_dna in results.keys():
                    continue

                counter += 1

                prev_dna = dna2str(model.dna[:i])
                # state is a dict of that hold all the saved outputs from previous steps for later use
                state = dict(prev_results[prev_dna])  # Must wrap in dict to make a copy
                results[current_dna] = self.pool.apply_async(self.perform_step, args=(i, model, state, train_step))

            print('Started ' + str(counter) + ' processes')
            # Wait for all processes from this section of dna to complete
            for dna, output in results.items():
                if isinstance(output, mp.pool.ApplyResult):
                    results[dna] = output.get()

            prev_results = results

        return results

    # @staticmethod
    # def prepare_step(i, model, state, train_step):
    #     if isinstance(train_step['func'], str):
    #         f_parts = train_step['func'].split('.')
    #         func = state[f_parts[0]]
    #         for part in f_parts:
    #             if hasattr(func, part):
    #                 func = getattr(func, part)
    #             else:
    #                 raise Exception('Could not get ' + train_step['func'] + ' from state')
    #     else:
    #         func = train_step['func']
    #     # get data with matching genes from previous stage of development and apply function + args of next gene
    #     args = (*[state[inp] for inp in train_step['inputs']], *model.dna[i])
    #     return partial(func, args)

    @staticmethod
    def perform_step(i, model, state, train_step):
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
                self.populate_next()

            # Get next fold of data for next generation
            x_train, x_test, y_train, y_test = next(self.data_fold_generator)
            results = self.learn(
                {'x_train': x_train, 'x_test': x_test, 'y_train': y_train}
            )
            
            self.score_fitness(results)

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


def generate_stratified_folds(data, n_splits, y_col):
    X = data.drop(y_col, axis=1)
    y = data[y_col]
    skf = StratifiedKFold(n_splits=n_splits)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield X_train, X_test, y_train, y_test



# below functions are for testing purposes


def do_nothing(data, n):
    return data


def add_cols(data, useless_variable):
    return data.sum(axis=1)


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error as mse
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
    Make ability to add steps before splitting data for ease of use with new data
    let functions change
    let there be nucleotides that represent continuous variables
    let mutation magnitude adjust during the run
"""
