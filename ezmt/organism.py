import asyncio
from copy import deepcopy
import inspect
import os
import json
import datetime
import importlib
import pandas as pd
import pickle

from ezmt.the_pickler import ThePickler, check_state_picklability


class Organism:

    def __init__(self, name, dna, parameters, knowledge=None, save_load_funcs=None, folder=None):
        self.name = name
        # self.dna represents the sequence of functions
        self.dna = dna if dna else []
        # self.parameters holds the arg values for the functions in self.dna
        self.parameters = parameters if parameters else {}
        # self.knowledge holds data generated from training that is needed for inference
        self.knowledge = knowledge if knowledge else {}
        # self.save_load_funcs holds custom saving and loading logic for state/knowledge objects
        self.save_load_funcs = save_load_funcs if save_load_funcs else {}
        if not folder:
            folder = f"organisms/{self.name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.folder = folder
        self.score = 0
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.dna == other.dna and self.parameters == other.parameters

    def __hash__(self):
        return hash(dna2str(self.dna) + str(self.parameters))

    def __str__(self):
        return dna2str(self.dna)

    def __repr__(self):
        return self.__str__()

    def add_gene(self, gene):
        self.dna.append(gene)

    async def run_gene(self, mode, gene_index, state, pool=None, log_state=False):
        if not self.dna[gene_index][mode]:
            return state

        is_async = self.is_gene_async(gene_index, 'train', state)
        run_in_parent_process = self.dna[gene_index][mode].get('run_in_parent_process', False)
        # TODO organisms with current step param run_in_parent_process=True should be sorted to the end of the organisms
        #  so that other async/parallel jobs can start first.
        #  Also it should be assigned False by default for train branch in config validation
        if is_async:
            # Async CPU
            new_state = await self.make_decision_async('train', gene_index, state)
        elif run_in_parent_process:
            new_state = self.make_decision('train', gene_index, state)
        elif pool:
            check_state_picklability(state)
            loop = asyncio.get_running_loop()
            new_state = await loop.run_in_executor(
                pool,
                self.make_decision, 'train', gene_index, state
            )
        else:
            new_state = await asyncio.to_thread(self.make_decision, 'train', gene_index, state)

        if log_state:
            folder = self.folder + f'/{mode}_log'
            self.save_state(folder, f"{gene_index}.json", new_state)
        return new_state

    def make_decision(self, mode, gene_index, state):
        # Synchronous decision-making logic
        func, args, kwargs, output_names = self._make_decision_common(
            mode, gene_index, state
        )
        if not func:
            return state
        output = func(*args, **kwargs)
        return self._update_state(state, output_names, output)

    async def make_decision_async(self, mode, gene_index, state):
        # Asynchronous decision-making logic
        func, args, kwargs, output_names = self._make_decision_common(
            mode, gene_index, state
        )
        if not func:
            return state
        output = await func(*args, **kwargs)
        return self._update_state(state, output_names, output)

    def _make_decision_common(self, mode, gene_index, state):
        # extract the right function, args, kwargs, and output names from the gene at index gene_index
        gene = self.dna[gene_index][
            mode
        ]  # training / inference version of current gene
        if not gene:
            return (
                None,
                None,
                None,
                None,
            )  # gene is inactive in this mode, return current state
        func = gene["func"]

        # TODO all string logic below this could be cleaned up. There is duplicate code, and args could be gotten from recusive getattr too

        if isinstance(
            func, str
        ):  # if str, get it from values of state ('model.run' => 'model' is a key in state)
            func = self.get_func_from_string(func, state)

        # get data with matching genes from previous stage of development and apply function + args of next gene
        args = []
        for arg in gene["args"]:
            if isinstance(arg, str):
                if arg in state:
                    args.append(state[arg])
                elif arg in self.parameters:
                    args.append(self.parameters[arg])
                elif arg.startswith('self.'):
                    attr_name = arg.split('.')[1]
                    if hasattr(self, attr_name):
                        args.append(getattr(self, attr_name))
            else:
                args.append(arg)

        kwargs = {}
        for key, val in gene["kwargs"].items():
            if isinstance(val, str):
                if val in state:
                    kwargs[key] = state[val]
                elif val in self.parameters:
                    kwargs[key] = self.parameters[val]
                elif val.startswith('self.'):
                    attr_name = val.split('.')[1]
                    if hasattr(self, attr_name):
                        kwargs[key] = getattr(self, attr_name)
            else:
                kwargs[key] = val

        output_names = gene["outputs"]  # output names
        return func, args, kwargs, output_names

    def get_func_from_string(self, func, state):
        f = func.split(".")
        func = state[f[0]]
        for part in f[1:]:
            if hasattr(func, part):
                func = getattr(func, part)
            else:
                raise Exception(f"Could not get {part} from {func}")
        return func

    def is_gene_async(self, gene_index, mode, state):
        func = self.dna[gene_index][mode]["func"]
        func = self.get_func_from_string(func, state) if isinstance(func, str) else func
        is_async = inspect.iscoroutinefunction(func)
        return is_async

    def _update_state(self, state, output_names, output):
        # Update State
        if output_names:
            if len(output_names) > 1:
                output = {o: output[j] for j, o in enumerate(output_names)}
            elif len(output_names) == 1:
                output = {o: output for o in output_names}
            state = {**state, **output}
        return state

    def mate(self, other):
        pass

    def reproduce(self):
        return Organism(self.name, deepcopy(self.dna), deepcopy(self.parameters), save_load_funcs=self.save_load_funcs)

    async def predict(self, x_new=None, log_states=False):
        folder = os.path.join(self.folder, "predictions", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        # TODO does this belong in the Organism class or the ModelTuner class?
        state = {**self.knowledge, "x_new": x_new}
        for gene_index in range(len(self.dna)):
            state = await self.run_gene("inference", gene_index, state, log_state=log_states)
        if "y_pred" in state:
            return state["y_pred"]
        else:
            raise Exception("No output found after last gene in the organism.")

    def save(self):
        create_folder(self.folder)

        # save dna
        formatted_dna, knowledge_to_save = self.create_formatted_dna()
        with open(f"{self.folder}/dna.json", "w") as f:
            json.dump(formatted_dna, f, indent=4)

        # save parameters
        with open(f"{self.folder}/parameters.json", "w") as f:
            json.dump(self.parameters, f, indent=4)

        # save state aka knowledge
        if self.knowledge:
            self.save_state(self.folder, "knowledge.json", self.knowledge)

        # save the custom saving and loading functions
        if self.save_load_funcs:
            with open(f"{self.folder}/save_load_funcs.json", "w") as f:
                json.dump(self.save_load_funcs, f, cls=ThePickler, folder=self.folder, indent=4)

    def save_state(self, folder, file_name, state):
        create_folder(folder)
        for key, val in state.items():
            # if there is a custom save function provided for this state object
            if key in self.save_load_funcs:
                # save it and replace the object in self.knowledge with the file name
                state[key] = self.save_load_funcs[key]['save'](folder, key, val)
        with open(f"{folder}/{file_name}", "w") as f:
            json.dump(state, f, cls=ThePickler, folder=folder, indent=4)

    def create_formatted_dna(self):
        """
        Format the DNA for saving. Get the knowledge from training that is necessary for inference.

        :return: formatted_dna, knowledge_to_save
        """
        dna_copy = []
        knowledge_to_save = {}
        available_inputs = ["x_new"]
        for step, gene in enumerate(self.dna):

            new_train = None
            if gene["train"]:
                # Don't save the function itself. Save a reference.
                train_func = gene["train"]["func"]
                if not isinstance(train_func, str):
                    train_func = get_function_reference(gene["train"]["func"])
                new_train = {**gene["train"], "func": train_func}

            new_inference = None
            if gene["inference"]:
                # Save args that come from training
                for inp in gene["inference"]["args"]:
                    if inp not in available_inputs and inp in self.knowledge:
                        knowledge_to_save[inp] = self.knowledge[inp]
                # Don't save the function itself. Save a reference.
                inf_func = gene["inference"]["func"]
                if isinstance(inf_func, str):
                    parent = inf_func.split(".")[0]
                    if parent not in available_inputs:
                        knowledge_to_save[parent] = self.knowledge[parent]
                else:
                    inf_func = get_function_reference(inf_func)
                available_inputs += gene["inference"]["outputs"]
                new_inference = {**gene["inference"], "func": inf_func}

            dna_copy.append(
                {"name": gene["name"], "train": new_train, "inference": new_inference}
            )

        return dna_copy, knowledge_to_save

    @classmethod
    def load(cls, name, folder):
        folder = f"organisms/{name}/{folder}"

        # Load custom saving and loading logic
        save_load_funcs = {}
        path = os.path.join(folder, "save_load_funcs.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                save_load_funcs = json.load(f)
            # Some knowledge is stored in pickle files. Load them
            for key, value in save_load_funcs.items():
                with open(os.path.join(folder, value), "rb") as pkl_file:
                    save_load_funcs[key] = pickle.load(pkl_file)

        # Load knowledge
        with open(os.path.join(folder, "knowledge.json"), "r") as f:
            knowledge = json.load(f)
        # Some knowledge is stored in other files. Load them
        for key, value in knowledge.items():
            if isinstance(value, str):
                if key in save_load_funcs:
                    knowledge[key] = save_load_funcs[key](folder, value)
                elif value.endswith(".pkl"):
                    with open(os.path.join(folder, value), "rb") as pkl_file:
                        knowledge[key] = pickle.load(pkl_file)
                elif value.endswith(".csv"):
                    knowledge[key] = pd.read_csv(value, index_col=0)

        # Load DNA
        with open(os.path.join(folder, "dna.json"), "r") as f:
            dna = json.load(f)

        # We don't save actual functions, just their references. We need to load them
        inference_outputs = []
        for gene in dna:
            # Train is not needed right now. Maybe in the future we will want to train more after saving and loading.
            if gene["inference"]:
                func_ref = gene["inference"]["func"]
                parent = func_ref.split(".")[0]
                if parent in knowledge or parent in inference_outputs:
                    continue
                gene["inference"]["func"] = load_function_from_reference(func_ref)

        # Load parameters
        with open(os.path.join(folder, "parameters.json"), "r") as f:
            parameters = json.load(f)

        return cls(name, dna, parameters, knowledge, save_load_funcs, folder)

    def reset(self):
        self.score = 0
        self.fitness = 0
        self.knowledge = {}


def get_function_reference(func):
    """Generate a string reference for a function, including nested paths."""
    module_name = func.__module__
    qualname = func.__qualname__
    return f"{module_name}.{qualname}"


def load_function_from_reference(func_ref):
    """Load a function from its string reference, handling nested paths."""
    module_name, *path = func_ref.split(".")
    module = importlib.import_module(module_name)
    func = module
    for part in path:
        func = getattr(func, part)

    return func


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def dna2str(dna):
    dna_str = ""
    for gene in dna:
        dna_str += gene["name"] + "("
        if gene["train"]:
            dna_str += ", ".join([str(a) for a in gene["train"]["args"]])
            for key, value in gene["train"]["kwargs"].items():
                dna_str += f", {key}={value}"
        dna_str += ")"
    return dna_str
