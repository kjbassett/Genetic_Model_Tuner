import asyncio
from copy import deepcopy
from functools import partial
import inspect
import os
import json
import datetime
import importlib
from typing import Iterable, Union
import pandas as pd
import pickle

from ezmt.the_pickler import ThePickler, check_state_picklability, load_frame
from ezmt.common_funcs import resolve_log_states

# How run_gene will execute a gene.
DISPATCH_AWAIT = "await"  # coroutine, awaited in the calling process
DISPATCH_PARENT = "parent"  # called inline in the calling process
DISPATCH_THREAD = "thread"  # worker thread, still the calling process
DISPATCH_SUBPROCESS = "subprocess"  # ProcessPoolExecutor; arguments are pickled

# The only dispatch that leaves the calling process, and so the only one that
# copies its arguments instead of sharing them by reference.
_COPYING_DISPATCHES = frozenset({DISPATCH_SUBPROCESS})


def resolve_dispatch(func, run_in_parent_process=False, has_pool=True):
    """Return how Organism.run_gene will execute ``func``.

    run_gene owns the decision, but callers need to predict it without running
    anything -- chiefly to reason about memory, since a gene sent to the process
    pool has its arguments pickled and therefore duplicated per caller. Keeping
    the rule here means run_gene and those callers cannot drift apart.

    Args:
        func: The resolved gene function. Must be the callable, not a string
            reference, since coroutine detection depends on the real object.
        run_in_parent_process: The gene's run_in_parent_process flag.
        has_pool: Whether a ProcessPoolExecutor was supplied to run_gene.

    Returns:
        One of DISPATCH_AWAIT, DISPATCH_PARENT, DISPATCH_THREAD or
        DISPATCH_SUBPROCESS.
    """
    if inspect.iscoroutinefunction(func):
        return DISPATCH_AWAIT
    if run_in_parent_process:
        return DISPATCH_PARENT
    if has_pool:
        return DISPATCH_SUBPROCESS
    return DISPATCH_THREAD


def copies_arguments(func, run_in_parent_process=False, has_pool=True):
    """Return whether running ``func`` as a gene would copy its arguments.

    Args:
        func: The resolved gene function.
        run_in_parent_process: The gene's run_in_parent_process flag.
        has_pool: Whether a ProcessPoolExecutor was supplied to run_gene.

    Returns:
        True when the gene crosses a process boundary and its arguments are
        pickled; False when it shares them by reference.
    """
    return (
        resolve_dispatch(func, run_in_parent_process, has_pool) in _COPYING_DISPATCHES
    )


class Organism:

    def __init__(
            self,
            name,
            dna,
            parameters,
            knowledge=None,
            save_load_funcs=None,
            version=None,
            gene_index=0,
            directory="organisms",
    ):
        self.name = name
        self.directory = directory
        # self.dna represents the sequence of functions
        self.dna = dna if dna else []
        # self.parameters holds the arg values for the functions in self.dna
        self.parameters = parameters if parameters else {}
        # self.knowledge holds data generated from training that is needed for inference
        self.knowledge = knowledge if knowledge else {}
        # self.save_load_funcs holds custom saving and loading logic for state/knowledge objects
        self.save_load_funcs = save_load_funcs if save_load_funcs else {}
        self.folder = self.new_version(version=version)
        self.score = 0
        self.fitness = 0
        self.gene_index = gene_index

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

    def new_version(self, name=None, version=None):
        if name is None:
            name = self.name
        if version is None:
            version = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.folder = f"{self.directory}/{name}/{version}"
        return self.folder

    def add_gene(self, gene):
        self.dna.append(gene)

    async def run_gene(self, mode, gene_index, state, pool=None, log_state=False):
        if not self.dna[gene_index][mode]:
            return state

        func, args, kwargs, output_names, run_in_parent_process = (
            self.get_gene_data(mode, gene_index, state)
        )

        if not func:
            return state
        # TODO organisms with current step param run_in_parent_process=True should be sorted to the end of the organisms
        #  so that other async/parallel jobs can start first.
        #  Also it should be assigned False by default for train branch in config validation
        #  Model tuner should create a task, not await run_gene
        dispatch = resolve_dispatch(func, run_in_parent_process, has_pool=bool(pool))
        if dispatch == DISPATCH_AWAIT:
            # Async CPU
            output = await func(*args, **kwargs)
        elif dispatch == DISPATCH_PARENT:
            output = func(*args, **kwargs)
        elif dispatch == DISPATCH_SUBPROCESS:
            check_state_picklability(state)
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(pool, partial(func, *args, **kwargs))
        else:
            output = await asyncio.to_thread(func, *args, **kwargs)

        new_state = self._update_state(state, output_names, output)

        if log_state:
            folder = self.folder + f"/{mode}_log"
            self.save_state(folder, f"{gene_index}.json", new_state)
        return new_state

    def get_gene_data(self, mode, gene_index, state):
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
                False,
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
                elif arg.startswith("self."):
                    attr_name = arg.split(".")[1]
                    if hasattr(self, attr_name):
                        args.append(getattr(self, attr_name))
                else:
                    args.append(arg)
            else:
                args.append(arg)

        kwargs = {}
        for key, val in gene["kwargs"].items():
            if isinstance(val, str):
                if val in state:
                    kwargs[key] = state[val]
                elif val in self.parameters:
                    kwargs[key] = self.parameters[val]
                elif val.startswith("self."):
                    attr_name = val.split(".")[1]
                    if hasattr(self, attr_name):
                        kwargs[key] = getattr(self, attr_name)
                else:
                    kwargs[key] = val
            else:
                kwargs[key] = val

        output_names = gene["outputs"]  # output names

        # Coroutine detection lives in resolve_dispatch so run_gene and any
        # caller predicting its behaviour read the same rule.
        run_in_parent_process = self.dna[gene_index][mode].get(
            "run_in_parent_process", False
        )
        return func, args, kwargs, output_names, run_in_parent_process

    def get_func_from_string(self, func, state):
        f = func.split(".")
        if f[0] in state:
            func = state[f[0]]
            for part in f[1:]:
                if hasattr(func, part):
                    func = getattr(func, part)
                else:
                    raise Exception(f"Could not get {part} from {func}")
        else:
            func = load_function_from_reference(func)
        return func

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
        return Organism(
            self.name,
            deepcopy(self.dna),
            deepcopy(self.parameters),
            save_load_funcs=self.save_load_funcs,
            directory=self.directory,
        )

    async def run(
            self,
            mode: str = "inference",
            data=None,
            log_states: Union[bool, int, Iterable[int]] = False,
            result_name="y_pred",
            update_knowledge: bool = False,
    ):
        # TODO does this belong in the Organism class or the ModelTuner class?
        state = {**self.knowledge, "x_new": data}
        states_to_log = resolve_log_states(log_states)
        while self.gene_index < len(self.dna):
            state = await self.run_gene(
                mode, self.gene_index, state, log_state=states_to_log is None or self.gene_index in states_to_log
            )
            self.gene_index += 1
        if update_knowledge:
            self.knowledge = {k: v for k, v in state.items() if k != "x_new"}
        if result_name in state:
            return state[result_name]
        else:
            raise Exception(
                f"No output named {result_name} found after last gene in the organism."
            )

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
                json.dump(
                    self.save_load_funcs,
                    f,
                    cls=ThePickler,
                    folder=self.folder,
                    indent=4,
                )

    def save_state(self, folder, file_name, state):
        create_folder(folder)
        state = dict(state)  # shallow copy. save_load_funcs only apply at top layer
        for key, val in state.items():
            # if there is a custom save function provided for this state object
            if key in self.save_load_funcs:
                # save it and replace the object in self.knowledge with the file name
                state[key] = self.save_load_funcs[key]["save"](folder, key, val)
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
                # Save kwargs that come from training (kwargs are resolved from state/
                # knowledge identically to args at run time in get_gene_data, so they
                # must be scanned identically here too)
                for inp in gene["inference"].get("kwargs", {}).values():
                    if isinstance(inp, str) and inp not in available_inputs and inp in self.knowledge:
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
    def load(cls, name, version: str = "latest", gene_index=None, directory="organisms"):
        if version == "latest":
            version = os.listdir(f"{directory}/{name}/")[-1]
        folder = f"{directory}/{name}/{version}"

        # Load DNA
        with open(os.path.join(folder, "dna.json"), "r") as f:
            dna = json.load(f)

        # load save_load_funcs for use in loading knowledge
        save_load_funcs = cls.load_save_load_funcs(folder)

        # load knowledge
        # if no gene_index is specified, load the fully trained model
        if gene_index is None:
            knowledge = cls.load_state(folder, "knowledge.json", save_load_funcs)
            gene_index = -1
        # if gene_index is an int, we load the knowledge of that index
        elif isinstance(gene_index, int):
            knowledge = cls.load_state(
                f"{folder}/train_log", f"{gene_index}.json", save_load_funcs
            )
        # if step is a string, find the index of the gene with the matching name, load knowledge of that index
        elif isinstance(gene_index, str):
            # get gene_index from gene name
            for i, gene in enumerate(dna):
                if gene["name"] == gene_index:
                    gene_index = i
                    break
            knowledge = cls.load_state(
                f"{folder}/train_log", f"{gene_index}.json", save_load_funcs
            )
        else:
            raise ValueError(f'gene_index must be None, int, or str. Got {gene_index} of type {type(gene_index)}')

        # We don't save dna functions, just their references. We need to load them
        # We assume that whoever is loading the organism has the same functions as when they created the dna
        inference_outputs = []
        for gene in dna:
            for mode in ["train", "inference"]:
                if gene[mode]:
                    func_ref = gene[mode]["func"]
                    parent = func_ref.split(".")[0]
                    # if the object was output by a previous step, assume that we will get it from state while running
                    if parent in knowledge or parent in inference_outputs:
                        continue
                    gene[mode]["func"] = load_function_from_reference(func_ref)

        # Load parameters
        with open(os.path.join(folder, "parameters.json"), "r") as f:
            parameters = json.load(f)

        return cls(
            name,
            dna,
            parameters,
            knowledge,
            save_load_funcs,
            version=version,
            gene_index=gene_index + 1,  # loaded knowdledge from gene_index, resume training at NEXT gene_index
            directory=directory,
        )

    @classmethod
    def load_save_load_funcs(cls, folder):
        # Load custom saving and loading logic
        save_load_funcs = {}
        path = os.path.join(folder, "save_load_funcs.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                save_load_funcs = json.load(f)
            # Some knowledge is stored in pickle files. Load them
            for key, save_load in save_load_funcs.items():
                for sl in ["save", "load"]:
                    if sl not in save_load:
                        continue
                    with open(os.path.join(folder, save_load[sl]), "rb") as pkl_file:
                        save_load_funcs[key][sl] = pickle.load(pkl_file)
        return save_load_funcs

    @classmethod
    def load_state(cls, folder, file_name, save_load_funcs):
        # Load knowledge
        with open(os.path.join(folder, file_name), "r") as f:
            knowledge = json.load(f)
        # Some knowledge is stored in other files. Load them
        for key, value in knowledge.items():
            if isinstance(value, str):
                if key in save_load_funcs and "load" in save_load_funcs[key]:
                    knowledge[key] = save_load_funcs[key]["load"](folder, value)
                elif value.endswith(".pkl"):
                    with open(os.path.join(folder, value), "rb") as pkl_file:
                        knowledge[key] = pickle.load(pkl_file)
                elif value.endswith((".parquet", ".csv")):
                    # .csv still handled so organisms saved before the switch
                    # to parquet keep loading.
                    knowledge[key] = load_frame(os.path.join(folder, value))
        return knowledge

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


def _resolve_reference(value, parameters):
    """Substitute a hyperparameter reference for its sampled value.

    Genes hold hyperparameters by name, not by value -- get_gene_data looks each
    one up in the organism's parameters at run time. Rendering the name would
    make two organisms with different hyperparameters produce the same string.

    Args:
        value: An arg or kwarg value from a gene, possibly a parameter name.
        parameters: The organism's sampled hyperparameters, or None to render
            references as-is.

    Returns:
        The sampled value when ``value`` names a hyperparameter, else ``value``.
    """
    if parameters and isinstance(value, str) and value in parameters:
        return parameters[value]
    return value


def dna2str(dna, parameters=None):
    """Render DNA as a string, used as the key for ezmt's per-gene result cache.

    Two organisms share a cached result only when their strings match up to that
    gene, so the string has to capture everything that makes their computations
    differ. That includes the hyperparameters each gene consumes: without
    ``parameters`` the genes render their parameter *names*, which are identical
    across the population, so every organism collapses onto one cache entry and
    only the first one ever runs.

    Passing ``parameters`` keeps the sharing that matters -- genes consuming no
    hyperparameters, or only ones pinned to a single value, still render
    identically and so still share a prefix -- while genes whose hyperparameters
    actually differ now fork.

    Args:
        dna: The organism's DNA, or a prefix slice of it.
        parameters: The organism's sampled hyperparameters. Omit only when the
            string is for display rather than for cache identity.

    Returns:
        The rendered DNA string.
    """
    dna_str = ""
    for gene in dna:
        dna_str += gene["name"] + "("
        if gene["train"]:
            dna_str += ", ".join(
                [str(_resolve_reference(a, parameters)) for a in gene["train"]["args"]]
            )
            for key, value in gene["train"]["kwargs"].items():
                dna_str += f", {key}={_resolve_reference(value, parameters)}"
        dna_str += ")"
    return dna_str
