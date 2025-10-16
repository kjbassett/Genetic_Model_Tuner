from copy import deepcopy
import inspect
import os
import json
import datetime
import importlib

import pandas as pd

from ezmt.the_pickler import ThePickler


class Organism:

    def __init__(self, name, dna, parameters, knowledge=None, folder=None):
        self.name = name
        # self.dna represents the sequence of functions
        self.dna = dna if dna else []
        # self.parameters holds the arg values for the functions in self.dna
        self.parameters = parameters if parameters else {}
        # self.knowledge holds data generated from training that is needed for inference
        self.knowledge = knowledge if knowledge else {}
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
            else:
                args.append(arg)

        kwargs = {}
        for key, val in gene["kwargs"].items():
            if isinstance(val, str):
                if val in state:
                    kwargs[key] = state[val]
                elif val in self.parameters:
                    kwargs[key] = self.parameters[val]
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
        return Organism(deepcopy(self.dna), deepcopy(self.parameters))

    async def predict(self, x_new=None, log_state=False):
        folder = os.path.join(self.folder, "predictions", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        # TODO does this belong in the Organism class or the ModelTuner class?
        state = {**self.knowledge, "x_new": x_new}
        for gene_index in range(len(self.dna)):
            if not self.dna[gene_index][
                "inference"
            ]:  # TODO organize make_decision and model_tuner.run_gene
                continue
            if self.is_gene_async(gene_index, "inference", state):
                state = await self.make_decision_async("inference", gene_index, state)
            else:
                state = self.make_decision("inference", gene_index, state)

            if log_state:
                create_folder(folder)
                json.dump(state, f, cls=ThePickler, folder=folder, indent=4)
        if "y_pred" in state:
            return state["y_pred"]
        else:
            raise Exception("No output found after last gene in the organism.")

    def save(self):
        # save params
        formatted_dna, knowledge_to_save = self.create_formatted_dna()
        create_folder(self.folder)
        with open(f"{self.folder}/dna.json", "w") as f:
            json.dump(formatted_dna, f, indent=4)
        with open(f"{self.folder}/parameters.json", "w") as f:
            json.dump(self.parameters, f, indent=4)
        with open(f"{self.folder}/knowledge.json", "w") as f:
            json.dump(knowledge_to_save, f, cls=ThePickler, folder=folder, indent=4)

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
        # Load knowledge
        with open(os.path.join(folder, "knowledge.json"), "r") as f:
            knowledge = json.load(f)
        # Some knowledge is stored in pickle files. Load them
        for key, value in knowledge.items():
            if isinstance(value, str):
                if value.endswith(".pkl"):
                    import pickle

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

        return cls(name, dna, parameters, knowledge, folder)

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


