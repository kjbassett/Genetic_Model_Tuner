# Easy Model Tuner (EZMT)

## What it does

This code is designed to optimize hyperparameters of a data analysis pipeline through a genetic algorithm.

The user defines choices of functions for each step of the pipeline as well as ranges of possible argument values for each function.
EZMT will "naturally select" the functions and arguments for each step through several generations. 
Each generation, the best will survive and repopulate with small mutations.
The best performing combination of functions and arguments at the end of the generations will be returned.

The user can save the returned model The user can also load the model later for inference.

## How to use it

### 1. Define a Model Space

A model space represents all possible configurations of your model. It's like notating all the possible combinations of DNA.
Here is an attempt at the explanation of the structure:
<details>

1. The outer list represents the whole model space, which holds every possible combination of options for the pipeline.
2. Each element in the model space list is a gene space, which is also a list.
Gene space i holds all the possible configurations for step i in the pipeline.
3. Each element in the gene space is a dictionary that contains 'name', 'train', 'inference' keys.
4. The name is the name of the option. 
5. The 'train' key holds a dictionary like this:
```
{
   'func': the function to be called,
   'args': [[list of possible discrete values for arg1], (start, end of continuous range of values for arg2)...],
   'kwargs': {
       "kwarg1": [list of possible discrete values for kwarg1], 
       "kwarg2": (start, end of continuous range of possible values for kwarg2)
   }
   'inputs': [list of inputs by name], # explained later
   'outputs': [list of outputs by name] # explained later
}
```
6. Inference has the same structure as train.
7. Train is used during training, inference is used during inference.
8. The output keys tell the model tuner to save the returned value(s) of the function in the "state".
    
    Ex: the function in the current step returns x, and the model space defines output: ['some_name'], 
    The state while running the tuner will include 'some_name': x after that step. 
9. The list of values in the 'input' key tells the model tuner to get a value by name from the state and use them as args
    In other words, a previous step's output can the current step's input
10. The 'func' key can be a string that corresponds to a key in the state, meaning that a previous step can output a function for a future step.
</details>

You (the user) can take some shortcuts when defining the model space.
Config validation will not only validate the config but also fix your shortcuts to standardize the structure.
Here is the logic it uses to do that.

<details>

* If the model space is a single dictionary, assume the whole pipeline is 1 step with 1 option.
* If an element in the model space list is a dictionary, assume it's a single choice and put it in a list,
* In a gene option, if only train is specified, the inference key will be created with a value of None. 
* Vice versa if only inference is specified.
* If the dictionary does not have 'train' or 'inference', 
    assume the same configuration for both train or inference.
    Copy the dict into the normal nested structure in both train and inference keys.
* If the dictionary does not have a 'name', try to get it from func.__name__ or func itself if func is a string
* If inputs is not in a list, put it in a list. Same for output
* If intputs or outputs keys are not present, create them with empty lists as values.
* If args not a list, put in list. If kwargs not a dict, put in dict.
* If the args/kwargs keys aren't present, they are created with an empty list/dict as their values.
</details>

Here is an example of a pipeline with 1 step that can either be logistic regression or ridge regressuib:

<details>

    model_space = [
        [
            {
                'name': init_logistic,
                'train': {
                    'func': sklearn.linear_model.LogisticRegression,
                    'args': [],
                    'kwargs': {
                        'penalty': ['l1', 'l2', 'elasticnet', None,
                        'tol': (0.00001, 0.01),
                        'C': (0.1, 10),
                        'solver': [‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’]
                    },
                    'inputs': [],
                    'outputs': 'model'
                },
                'inference': {
                    'func': some_function_that_inits_and_applies_saved_params_from_training
                    TODO fill me in
                }
            },
            {
                TODO Fill me in with ridge regression init
            },
        ],
        [
            TODO fill me in with .fit 
        ]
    ]

</details>

### 2. Instantiate The Model Tuner

    mt = ModelTuner(
        model_space, data, y_col, generations=1, pop_size=20, goal='min'
    )

### 3. Run The Model Tuner

    model = mt.run()

### 4. Save The Returned Model

    model.save()

### 5. Load the model for inference

    TBD
