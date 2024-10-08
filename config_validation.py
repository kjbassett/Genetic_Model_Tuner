from copy import deepcopy


def validate_config(model_space):
    new_model_space = []
    names_seen = {}

    if isinstance(model_space, dict): # One choice of function
        model_space = [model_space]  # Standardize

    for gene_space in model_space:
        if isinstance(gene_space, dict):  # only one choice of function
            gene_space = [gene_space]  # Standardize. Could be a list of functions

        new_gene_space = []
        for function_dict in gene_space:
            function_dict = validate_function_dict(function_dict, names_seen)
            new_gene_space.append(function_dict)
        new_model_space.append(new_gene_space)

    return new_model_space


def validate_function_dict(function_dict, names_seen):
    if not isinstance(function_dict, dict):
        raise ValueError("The third layer in should be a dictionary. Got " + str(type(function_dict)))
    validate_name(function_dict, names_seen)
    function_dict = validate_structure(function_dict)  # Can transform structure
    validate_info(function_dict['train'])
    validate_info(function_dict['inference'])
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


def validate_info(function_dict):
    if function_dict is None:
        return
    if not isinstance(function_dict, dict):
        raise TypeError(f"{function_dict} is not a dictionary.")
    if 'func' not in function_dict:
        raise ValueError(f"No 'func' key found in {function_dict}.")
    if not callable(function_dict['func']) and not isinstance(function_dict['func'], str):
        raise TypeError(f"'func' value in {function_dict} is not callable or string.")
    for io in ('inputs', 'outputs'):
        if io not in function_dict:
            function_dict[io] = []
            continue
        if isinstance(function_dict[io], str):
            function_dict[io] = [function_dict[io]]  # standardize
        if not hasattr(function_dict[io], '__iter__'):
            raise TypeError(f"'{io}' value in {function_dict} is not iterable.")
    if 'args' not in function_dict:
        function_dict['args'] = []
    else:
        if not isinstance(function_dict['args'], list):
            raise TypeError(f"'args' value in {function_dict} is not a list.")
        for arg in function_dict['args']:
            if not hasattr(arg, '__iter__'):
                raise TypeError(f"Argument {arg} in 'args' value of {function_dict} is not iterable.")
    if 'kwargs' not in function_dict:
        function_dict['kwargs'] = dict()
    if not isinstance(function_dict['kwargs'], dict):
        raise TypeError(f"'kwargs' value in {function_dict} is not a dictionary.")
    for value in function_dict['kwargs'].values():
        if not hasattr(value, '__iter__'):
            raise TypeError(f"Value {value} in 'kwargs' value of {function_dict} is not iterable.")
