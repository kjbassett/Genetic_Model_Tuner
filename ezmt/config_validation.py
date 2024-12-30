from copy import deepcopy

from ezmt.hyperparameters import Hyperparameter


def validate_config(model_space, hyperparams):
    validate_hyperparams(hyperparams)

    names_seen = {}
    available_args = {"x_train", "x_test", "y_train", "y_test"}  # To track the objects available to be args (or func)
    available_args.update(hyperparams.keys())  # Add hyperparameters to the available_args

    if isinstance(model_space, dict):  # One choice of function
        model_space = [model_space]  # Standardize format

    for ti in ('train', 'inference'):  # Must do all train first because its outputs can be used in inference
        # x_train, x_test, y_train, y_test are available during training only
        if ti == 'inference':
            available_args.difference({"x_train", "x_test", "y_train", "y_test"})
            available_args.update({"x_new"})
        for gs_idx, gene_space in enumerate(model_space):
            if isinstance(gene_space, dict):  # user supplied only one choice of function
                gene_space = [gene_space]  # Standardize. Could have been a list of functions
            for fd_idx, function_dict in enumerate(gene_space):
                if ti == 'train':  # we only need to validate the outer dict once
                    gene_space[fd_idx] = validate_outer_function_dict(function_dict, names_seen)
                gene_space[fd_idx][ti] = validate_inner_function_dict(function_dict[ti], available_args)
                print(gs_idx, fd_idx, ti)
                print(available_args)
            model_space[gs_idx] = gene_space  # replace the original list with the validated one
    return model_space


def validate_hyperparams(hyperparams):
    for name, hp in hyperparams.items():
        if isinstance(hp, Hyperparameter):
            continue  # Valid
        elif hasattr(hp, '__iter__') and not isinstance(hp, str):
            continue  # Valid if it's an iterable (discrete options)
        else:
            raise TypeError(
                f"Each hyperparam must be of type Hyperparameter or a non-string iterable. "
                f"Invalid element: {name}. "
                f"Type: {type(hp)}"
            )


def validate_outer_function_dict(outer_dict, names_seen):
    """
    validates and standardizes the function dictionary, containing both train and inference
    :param outer_dict:
    :param names_seen:
    :param previous_outputs:
    :return: function_dict with standardized structure
    """
    if not isinstance(outer_dict, dict):
        raise ValueError("The third layer in should be a dictionary. Got " + str(type(outer_dict)))
    validate_name(outer_dict, names_seen)
    outer_dict = validate_structure(outer_dict)  # Can transform structure
    return outer_dict


def validate_name(function_dict, names_seen):
    # Make function_dict (AKA potential training step) is given a name
    # TODO Make this less repetitive
    # Check in outer layer
    if 'name' not in function_dict:
        # if name not provided and one function for training/inference, we can use the name of the function
        if 'func' in function_dict:
            if callable(function_dict['func']):
                function_dict['name'] = function_dict['func'].__name__
            else:  # function is a string representing the output of a previous step. Function stored in the state
                function_dict['name'] = function_dict['func']

        # Check in inner layers
        elif 'train' in function_dict and function_dict['train']:
            if 'name' in function_dict['train']:
                function_dict['name'] = function_dict['train']['name']
            elif callable(function_dict['train']['func']):
                function_dict['name'] = function_dict['train']['func'].__name__
            else:  # function is a string representing the output of a previous step. Function stored in the state
                function_dict['name'] = function_dict['train']['func']
        elif 'inference' in function_dict and function_dict['inference']:
            if 'name' in function_dict['train']:
                function_dict['name'] = function_dict['inference']['name']
            elif callable(function_dict['inference']['func']):
                function_dict['name'] = function_dict['inference']['func'].__name__
            else:  # function is a string representing the output of a previous step. Function stored in the state
                function_dict['name'] = function_dict['inference']['func']
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


def validate_inner_function_dict(function_dict, available_args):
    if function_dict is None:
        return
    if not isinstance(function_dict, dict):
        raise TypeError(f"{function_dict} is not a dictionary.")

    if 'func' not in function_dict:
        raise ValueError(f"No 'func' key found in {function_dict}.")

    # Check for GPU flag
    function_dict['gpu'] = function_dict.get('gpu', False)

    # Check if 'func' is a string and validate it against previous outputs
    if isinstance(function_dict['func'], str):
        needed_output = function_dict['func'].split(".")[0]
        if needed_output not in available_args:
            raise ValueError(f"Function reference '{needed_output}' not found in any previous outputs.")
    elif not callable(function_dict['func']):
        raise TypeError(f"'func' value in {function_dict} is not callable or a valid string reference.")

    # Validate args and kwargs
    function_dict['args'] = validate_args(function_dict.get('args', []), available_args)
    function_dict['kwargs'] = validate_kwargs(function_dict.get('kwargs', {}), available_args)

    # Validate outputs
    function_dict['outputs'] = validate_output(function_dict.get('outputs', []))
    # Add the current step's outputs to previous_outputs for future input checks
    available_args.update(function_dict['outputs'])

    return function_dict


def validate_output(io_list):
    if isinstance(io_list, str):
        io_list = [io_list]  # Wrap single string into a list
    if not isinstance(io_list, list):
        raise TypeError(f"'outputs' must be a string or a list.")

    # Validate that all elements in the list are strings
    for element in io_list:
        if not isinstance(element, str):
            raise TypeError(f"Each element in 'outputs' must be a string. Invalid element: {element}")

    return io_list


def validate_args(args_list, available_args):
    if not isinstance(args_list, (list, tuple)):
        args_list = [args_list]  # Wrap single value into a list


    for arg in args_list:
        # If we're validating inputs, ensure they are valid previous outputs or predefined inputs
        if isinstance(arg, str):
            if arg not in available_args:
                print(
                    f"Warning: arg '{arg}' is not an available arg. "
                    f"You can ignore this if the provided function takes '{arg}' as a string argument."
                )
    return args_list


def validate_kwargs(kwargs_dict, available_args):
    if not isinstance(kwargs_dict, dict):
        raise TypeError(f"'kwargs' must be a dictionary.")

    for key, value in kwargs_dict.items():
        if isinstance(value, str) and value not in available_args:
            print(
                f"Warning: kwarg referencing '{value}' is not an available argument. "
                f"You can ignore this if the provided function takes '{value}' as a string argument."
            )

    return kwargs_dict

