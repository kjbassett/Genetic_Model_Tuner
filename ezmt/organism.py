from copy import deepcopy
import os
import json
import datetime
import importlib


class Organism:

    def __init__(self, dna=None, knowledge=None):
        if dna is None:
            self.dna = []
        else:
            self.dna = dna
        if knowledge is None:
            self.knowledge = {}
        else:
            self.knowledge = knowledge
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

    def __repr__(self):
        return self.__str__()

    def add_gene(self, gene):
        self.dna.append(gene)
    
    def make_decision(self, mode, gene_index, state):
        # Synchronous decision-making logic
        func, args, gene = self._make_decision_common(mode, gene_index, state)
        if not func:
            return state
        output = func(*args, **gene['kwargs'])
        return self._update_state(state, gene, output)

    async def make_decision_async(self, mode, gene_index, state):
        # Asynchronous decision-making logic
        func, args, gene = self._make_decision_common(mode, gene_index, state)
        if not func:
            return state
        output = await func(*args, **gene['kwargs'])
        return self._update_state(state, gene, output)

    def _make_decision_common(self, mode, gene_index, state):
        if not self.dna[gene_index][mode]:
            return None, None, None  # gene is inactive in this mode, return current state
        gene = self.dna[gene_index][mode]  # training version of current gene
        func = gene['func']

        if isinstance(func, str):  # if str, get it from values of state ('model.run' => 'model' is a key in state)
            f = func.split('.')
            func = state[f[0]]
            for part in f[1:]:
                if hasattr(func, part):
                    func = getattr(func, part)
                else:
                    raise Exception(f'Could not get {part} from {func}')

        # get data with matching genes from previous stage of development and apply function + args of next gene
        args = (state[arg] if isinstance(arg, str) and arg in state else arg for arg in gene['args'])
        return func, args, gene

    def _update_state(self, state, gene, output):
        # Update State
        ons = gene['outputs']  # output names
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

    def predict(self, x_new):
        # TODO does this belong in the Organism class or the ModelTuner class?
        state = {**self.knowledge, 'x_new': x_new}
        for gene_index in range(len(self.dna)):
            state = self.make_decision('inference', gene_index, state)
        if 'y_pred' in state:
            return state['y_pred']
        else:
            raise Exception('No output found after last gene in the organism.')

    def save(self, name):
        if not os.path.exists('../organisms'):
            os.makedirs('../organisms')
        if not os.path.exists(f"organisms/{name}"):
            os.makedirs(f"organisms/{name}")
        folder = f"organisms/{name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(folder)

        # save params
        formatted_dna, knowledge_to_save = self.create_formatted_dna()
        with open(f"{folder}/dna.json", "w") as f:
            json.dump(formatted_dna, f, indent=4)
        with open(f"{folder}/knowledge.json", "w") as f:
            json.dump(knowledge_to_save, f, cls=ThePickler, folder=folder, indent=4)

    def create_formatted_dna(self):
        """
        Format the DNA for saving. Get the knowledge from training that is necessary for inference.

        :return: formatted_dna, knowledge_to_save
        """
        dna_copy = []
        knowledge_to_save = {}
        available_inputs = ['x_new']
        for step, gene in enumerate(self.dna):

            new_train = None
            if gene['train']:
                # Don't save the function itself. Save a reference.
                train_func = gene['train']['func']
                if not isinstance(train_func, str):
                    train_func = get_function_reference(gene['train']['func'])
                new_train = {**gene['train'], 'func': train_func}

            new_inference = None
            if gene['inference']:
                # Save args that come from training
                for inp in gene['inference']['args']:
                    if inp not in available_inputs and inp in self.knowledge:
                        knowledge_to_save[inp] = self.knowledge[inp]
                # Don't save the function itself. Save a reference.
                inf_func = gene['inference']['func']
                if isinstance(inf_func, str):
                    parent = inf_func.split('.')[0]
                    if parent not in available_inputs:
                        knowledge_to_save[parent] = self.knowledge[parent]
                else:
                    inf_func = get_function_reference(inf_func)
                available_inputs += gene['inference']['outputs']
                new_inference = {**gene['inference'], 'func': inf_func}

            dna_copy.append({
                'name': gene['name'],
                'train': new_train,
                'inference': new_inference
            })

        return dna_copy, knowledge_to_save

    @classmethod
    def load(cls, folder):
        # Load knowledge
        with open(os.path.join(folder, "knowledge.json"), "r") as f:
            knowledge = json.load(f)
        # Some knowledge is stored in pickle files. Load them
        for key, value in knowledge.items():
            if value.endswith('.pkl'):
                import pickle
                with open(os.path.join(folder, value), 'rb') as pkl_file:
                    knowledge[key] = pickle.load(pkl_file)

        # Load DNA
        with open(os.path.join(folder, "dna.json"), "r") as f:
            dna_loaded = json.load(f)
        # We don't save actual functions, just their references. We need to load them
        inference_outputs = []
        for gene in dna_loaded:
            # Train is not needed right now. Maybe in the future we will want to train more after saving and loading.
            if gene['inference']:
                func_ref = gene['inference']['func']
                parent = func_ref.split('.')[0]
                if parent in knowledge or parent in inference_outputs:
                    continue
                gene['inference']['func'] = load_function_from_reference(func_ref)

        return cls(dna_loaded, knowledge)

    def reset(self):
        self.score = 0
        self.fitness = 0
        self.knowledge = None


def get_function_reference(func):
    """Generate a string reference for a function, including nested paths."""
    module_name = func.__module__
    qualname = func.__qualname__
    return f"{module_name}.{qualname}"


def load_function_from_reference(func_ref):
    """Load a function from its string reference, handling nested paths."""
    module_name, *path = func_ref.split('.')
    module = importlib.import_module(module_name)
    func = module
    for part in path:
        func = getattr(func, part)

    return func


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
