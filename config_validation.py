from copy import deepcopy


"""
1. The outer list holds the whole model space
2. Each element in the model space represents a step during processing.
3. The model space's elements are lists.
4. Those lists hold all the possible choices for that step in the model.
5. Each choice is a dictionary that contains 'name', 'train', 'inference' keys.
6. The name is the name of the option. 
7. Train holds a dictionary with a function, argument choices, kwarg choices, and inputs/outputs.
8. Inference has the same structure as train.
9. Train is used during training, inference is used during inference.

If the model space is a single dictionary, assume 1 step with 1 choice.
If an element in the model space is a dictionary, assume it's a single choice and put it in a list.
If only train is specified, inference will be skipped 
If only inference is specified, training will skip this step.
If the dictionary does not have 'train' or 'inference', 
    Assume function info in outer dict and copy to nested structure in both train and inference.
If the dictionary does not have a 'name', try to get it from func.__name__ or func itself if func is a string
If inputs or outputs are not in lists, put them in lists
If args not a list, put in list. If kwargs not a dict, put in dict.


The model tuner will choose functions and arguments for each step and use the train version for training.
and the best will survive and repopulate with small mutations.
After many generations, the best combination of functions and arguments will be saved.
Based on the saved file, the user can load the inference version of the model for use on new data.
"""

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
