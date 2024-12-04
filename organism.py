from copy import deepcopy
import os
import json
import datetime


class Organism:

    def __init__(self, dna=None):
        if dna is None:
            self.dna = []
        else:
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
    
    def make_decision(self, gene_index, state):
        gene_train = self.dna[gene_index]['train']  # training version of current gene
        func = gene_train['func']

        if isinstance(func, str):  # if str, get it from values of state ('model.run' => 'model' is a key in state)
            f = func.split('.')
            func = state[f[0]]
            for part in f[1:]:
                if hasattr(func, part):
                    func = getattr(func, part)
                else:
                    raise Exception('Could not get ' + part + ' from ' + func)

        # get data with matching genes from previous stage of development and apply function + args of next gene
        args = (*[state[inp] for inp in gene_train['inputs']], *gene_train['args'])
        output = func(*args, **gene_train['kwargs'])

        # Update State
        ons = gene_train['outputs']  # output names
        if ons:
            if len(ons) > 1:
                output = {o: output[j] for j, o in enumerate(ons)}
            elif len(ons) == 1:
                output = {o: output for o in ons}
            state = {**state, **output}

        return state

    def mate(self, other):
        pass

    def reproduce(self):
        return Organism(deepcopy(self.dna))

    def save_params(self, name):
        if self.knowledge is None:
            raise ValueError("Knowledge is not available. Cannot save parameters.")
        if not os.path.exists('organisms'):
            os.makedirs('organisms')
        if not os.path.exists(f"organisms/{name}"):
            os.makedirs(f"organisms/{name}")
        folder = f"organisms/{name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(folder)
        params_needed = self._find_params_to_save()
        params = {param: self.knowledge[param] for param in params_needed}

        with open(f"{folder}/config.json", "w") as f:
            json.dump(params, f, cls=ThePickler, folder=folder, indent=4)

    def _find_params_to_save(self):
        params_needed = []
        for gene in self.dna[::-1]:
            if gene['inference']:
                for out in gene['inference']['outputs']:
                    if out in params_needed:
                        params_needed.remove(out)
                for inp in gene['inference']['inputs']:
                    if inp not in ['x_train', 'y_train', 'x_test', 'y_test']:
                        params_needed.append(inp)
        return params_needed

    def reset(self):
        self.score = 0
        self.fitness = 0
        self.knowledge = None


def dna2str(dna):
    dna_str = ''
    for gene in dna:
        dna_str += gene['name'] + '('
        if gene['train']:
            dna_str += ', '.join([str(a) for a in gene['train']['args']])
            for key, value in gene['train']['kwargs'].items():
                dna_str += f', {key}={value}'
        dna_str += ')'
    return dna_str


class ThePickler(json.JSONEncoder):
    def __init__(self, folder='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            import pickle
            try:
                pickled_data = pickle.dumps(obj)
                file_name = f"{id(obj)}.pkl"
                with open(os.path.join(self.folder, file_name), 'wb') as f:
                    f.write(pickled_data)
                return file_name
            except Exception as e:
                print(f"Error pickling object: {str(e)}")
                return str(obj)
