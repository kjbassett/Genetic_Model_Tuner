from copy import deepcopy
import random


class ContinuousRange:
    def __init__(self, start, end):
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"Start and end values for ContinuousRange must be numeric. Got {start}, {end}")
        if start >= end:
            raise ValueError(f"Start value must be less than end value. Got start={start} and end={end}")
        self.start = start
        self.end = end

    def __repr__(self):
        return f"ContinuousRange({self.start}, {self.end})"

    def sample(self):
        """Sample a random value from the continuous range."""
        return random.uniform(self.start, self.end)


def validate_config(model_space):
    new_model_space = []
    names_seen = {}
    previous_outputs = set()  # To track the outputs seen so far

    if isinstance(model_space, dict): # One choice of function
        model_space = [model_space]  # Standardize

    for gene_space in model_space:
        if isinstance(gene_space, dict):  # only one choice of function
            gene_space = [gene_space]  # Standardize. Could be a list of functions

        new_gene_space = []
        for function_dict in gene_space:
            function_dict = validate_function_dict(function_dict, names_seen, previous_outputs)
            new_gene_space.append(function_dict)
            # Add the current step's outputs to previous_outputs for future reference
            previous_outputs.update(function_dict['train']['outputs'])
            previous_outputs.update(function_dict['inference']['outputs'])
        new_model_space.append(new_gene_space)

    return new_model_space


def validate_function_dict(function_dict, names_seen, previous_outputs):
    if not isinstance(function_dict, dict):
        raise ValueError("The third layer in should be a dictionary. Got " + str(type(function_dict)))
    validate_name(function_dict, names_seen)
    function_dict = validate_structure(function_dict)  # Can transform structure
    validate_info(function_dict['train'], previous_outputs)
    validate_info(function_dict['inference'], previous_outputs)
    return function_dict


def validate_name(function_dict, names_seen):
    # Make function_dict (AKA potential training step) is given a name
    if 'name' not in function_dict:
        # if name not provided and one function for training/inference, we can use the name of the function it
        if 'func' in function_dict:
            if callable(function_dict['func']):
                function_dict['name'] = function_dict['func'].__name__
            else:  # function is a string representing the output of a previous step. Function stored in the state
                function_dict['name'] = function_dict['func']
        else:
            raise ValueError(
                "No name provided and no function to pull name from."
            )
    # If multiple function dicts with the same name, add counter to name to make it unique
    if function_dict['name'] in names_seen:
        names_seen[function_dict['name']] += 1
        function_dict['name'] += f'_{names_seen[function_dict["name"]]}'
    else:
        names_seen[function_dict['name']] = 0


def validate_structure(function_dict):
    # different function info for training and inference. function infos in 'train' and 'inference' keys
    if 'train' in function_dict or 'inference' in function_dict:
        if 'func' in function_dict:
            msg = "Bad function dictionary config. "
            msg += "You should only use 'func' key  when train and inference functions are the same."
            raise ValueError(msg)
        # validate train func
        if 'train' not in function_dict:
            function_dict['train'] = None
        # validate inference func
        if 'inference' not in function_dict:
            function_dict['inference'] = None

    elif 'func' not in function_dict:
        raise ValueError(f"Bad function dictionary config. No 'train', 'inference' or 'func' key found.")

    else:  # Only one function provided for training/inference, assume it's for both
        function_dict = {'train': function_dict, 'inference': deepcopy(function_dict), 'name': function_dict['name']}
    return function_dict


def validate_info(function_dict, previous_outputs):
    if function_dict is None:
        return
    if not isinstance(function_dict, dict):
        raise TypeError(f"{function_dict} is not a dictionary.")
    if 'func' not in function_dict:
        raise ValueError(f"No 'func' key found in {function_dict}.")

    # Check if 'func' is a string and validate it against previous outputs
    if isinstance(function_dict['func'], str):
        if function_dict['func'] not in previous_outputs:
            raise ValueError(f"Function reference '{function_dict['func']}' not found in any previous outputs.")
    elif not callable(function_dict['func']):
        raise TypeError(f"'func' value in {function_dict} is not callable or a valid string reference.")

    # Validate inputs and outputs
    function_dict['inputs'] = validate_io(function_dict.get('inputs', []), 'inputs')
    function_dict['outputs'] = validate_io(function_dict.get('outputs', []), 'outputs')

    # Validate args and kwargs
    function_dict['args'] = validate_args(function_dict.get('args', []))
    function_dict['kwargs'] = validate_kwargs(function_dict.get('kwargs', {}))

    return function_dict


def validate_io(io_list, io_type):
    if isinstance(io_list, str):
        io_list = [io_list]  # Wrap single string into a list
    if not isinstance(io_list, list):
        raise TypeError(f"'{io_type}' must be a string or a list.")

    # Validate that all elements in the list are strings
    for element in io_list:
        if not isinstance(element, str):
            raise TypeError(f"Each element in '{io_type}' must be a string. Invalid element: {element}")

    return io_list


def validate_args(args_list):
    if not isinstance(args_list, list):
        raise TypeError(f"'args' must be a list.")

    for arg in args_list:
        if isinstance(arg, ContinuousRange):
            continue  # Valid if it's a ContinuousRange
        elif hasattr(arg, '__iter__'):
            continue  # Valid if it's an iterable (discrete options)
        else:
            raise TypeError(f"Each element in 'args' must be a ContinuousRange or an iterable. Invalid element: {arg}")

    return args_list


def validate_kwargs(kwargs_dict):
    if not isinstance(kwargs_dict, dict):
        raise TypeError(f"'kwargs' must be a dictionary.")

    for key, value in kwargs_dict.items():
        if isinstance(value, ContinuousRange):
            continue  # Valid if it's a ContinuousRange
        elif hasattr(value, '__iter__'):
            continue  # Valid if it's an iterable (discrete options)
        else:
            raise TypeError(
                f"Each value in 'kwargs' must be a ContinuousRange or an iterable. Invalid element: {value}")

    return kwargs_dict

