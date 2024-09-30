import unittest
from config_validation import validate_config
from copy import deepcopy


class TestValidateConfig(unittest.TestCase):

    def test_validate_config_empty_list(self):
        model_space = []
        expected_output = []
        self.assertEqual(validate_config(model_space), expected_output)

    def test_validate_config_single_dict(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'inputs': 'input',
            'outputs': 'output'
        }
        expected_output = [[{
            'name': 'test_func',
            'train': {
                'name': 'test_func',
                'func': model_space['func'],
                'inputs': ['input'],
                'outputs': ['output'],
                'args': [],
                'kwargs': {}
            },
            'inference': {
                'name': 'test_func',
                'func': model_space['func'],
                'inputs': ['input'],
                'outputs': ['output'],
                'args': [],
                'kwargs': {}
            }
        }]]
        result = validate_config(model_space)
        self.assertEqual(result, expected_output)

    def test_validate_config_multiple_dicts(self):
        model_space = [
            {
                'name': 'test_func1',
                'func': lambda x: x,
                'inputs': 'input1',
                'outputs': 'output1'
            },
            {
                'name': 'test_func2',
                'func': lambda y: y,
                'inputs': 'input2',
                'outputs': 'output2'
            }
        ]
        expected_output = [[{
            'name': 'test_func1',
            'train': {
                'name': 'test_func1',
                'func': model_space[0]['func'],
                'inputs': ['input1'],
                'outputs': ['output1'],
                'args': [],
                'kwargs': {}
            },
            'inference': {
                'name': 'test_func1',
                'func': model_space[0]['func'],
                'inputs': ['input1'],
                'outputs': ['output1'],
                'args': [],
                'kwargs': {}
            }
        }], [{
            'name': 'test_func2',
            'train': {
                'name': 'test_func2',
                'func': model_space[1]['func'],
                'inputs': ['input2'],
                'outputs': ['output2'],
                'args': [],
                'kwargs': {}
            },
            'inference': {
                'name': 'test_func2',
                'func': model_space[1]['func'],
                'inputs': ['input2'],
                'outputs': ['output2'],
                'args': [],
                'kwargs': {}
            }
        }]]
        result = validate_config(model_space)
        self.assertEqual(result, expected_output)

    def test_validate_config_with_function_choice(self):
        model_space = [
            [
                {'name': 'step1', 'func': 'func1', 'inputs': 'input1', 'outputs': 'output1'},
                {'name': 'step2', 'func': 'func2', 'inputs': ['input2'], 'outputs': ['output2']}
            ],
            [
                {'name': 'step3', 'func': 'func3', 'inputs': 'input3', 'outputs': 'output3'}
            ]
        ]

        expected_output = [
            [
                {'name': 'step1',
                 'train': {'name': 'step1', 'func': 'func1', 'inputs': ['input1'], 'outputs': ['output1'], 'args': [],
                           'kwargs': {}},
                 'inference': {'name': 'step1', 'func': 'func1', 'inputs': ['input1'], 'outputs': ['output1'],
                               'args': [], 'kwargs': {}}},
                {'name': 'step2',
                 'train': {'name': 'step2', 'func': 'func2', 'inputs': ['input2'], 'outputs': ['output2'], 'args': [],
                           'kwargs': {}},
                 'inference': {'name': 'step2', 'func': 'func2', 'inputs': ['input2'], 'outputs': ['output2'],
                               'args': [], 'kwargs': {}}}
            ],
            [
                {'name': 'step3',
                 'train': {'name': 'step3', 'func': 'func3', 'inputs': ['input3'], 'outputs': ['output3'], 'args': [],
                           'kwargs': {}},
                 'inference': {'name': 'step3', 'func': 'func3', 'inputs': ['input3'], 'outputs': ['output3'],
                               'args': [], 'kwargs': {}}}
            ]
        ]

        result = validate_config(model_space)
        self.assertEqual(result, expected_output)

    def test_validate_config_nested_too_deep(self):
        model_space = [
            [
                [
                    {'name': 'func1', 'func': lambda x: x, 'inputs': 'input1', 'outputs': 'output1'},
                    {'name': 'func2', 'func': lambda y: y, 'inputs': 'input2', 'outputs': 'output2'}
                ],
                [
                    {'name': 'func3', 'func': lambda z: z, 'inputs': 'input3', 'outputs': 'output3'}
                ]
            ],
            [
                {'name': 'func4', 'func': lambda a: a, 'inputs': 'input4', 'outputs': 'output4'}
            ]
        ]
        with self.assertRaises(ValueError) as context:
            validate_config(model_space)
        print(context.exception)
        assert "The third layer in should be a dictionary. Got <class 'list'>" in str(context.exception)

    def test_validate_config_mixed_list(self):
        model_space = [
            {
                'name': 'func1',
                'func': lambda x: x,
                'inputs': 'input1',
                'outputs': 'output1'
            },
            [
                {
                    'name': 'func2',
                    'func': lambda x: x,
                    'inputs': 'input2',
                    'outputs': 'output2'
                },
                {
                    'name': 'func3',
                    'func': lambda x: x,
                    'inputs': 'input3',
                    'outputs': 'output3'
                }
            ]
        ]
        expected_output = [
            [{
                'name': 'func1',
                'train': {
                    'name': 'func1',
                    'func': model_space[0]['func'],
                    'inputs': ['input1'],
                    'outputs': ['output1'],
                    'args': [],
                    'kwargs': {}
                },
                'inference': {
                    'name': 'func1',
                    'func': model_space[0]['func'],
                    'inputs': ['input1'],
                    'outputs': ['output1'],
                    'args': [],
                    'kwargs': {}
                }
            }],
            [
                {
                    'name': 'func2',
                    'train': {
                        'name': 'func2',
                        'func': model_space[1][0]['func'],
                        'inputs': ['input2'],
                        'outputs': ['output2'],
                        'args': [],
                        'kwargs': {}
                    },
                    'inference': {
                        'name': 'func2',
                        'func': model_space[1][0]['func'],
                        'inputs': ['input2'],
                        'outputs': ['output2'],
                        'args': [],
                        'kwargs': {}
                    }
                },
                {
                    'name': 'func3',
                    'train': {
                        'name': 'func3',
                        'func': model_space[1][1]['func'],
                        'inputs': ['input3'],
                        'outputs': ['output3'],
                        'args': [],
                        'kwargs': {}
                    },
                    'inference': {
                        'name': 'func3',
                        'func': model_space[1][1]['func'],
                        'inputs': ['input3'],
                        'outputs': ['output3'],
                        'args': [],
                        'kwargs': {}
                    }
                }
            ]
        ]
        result = validate_config(model_space)
        self.assertEqual(result, expected_output)

    def test_validate_config_updates_names_seen(self):
        model_space = [
            {'name': 'func1', 'func': lambda x: x, 'inputs': 'input1', 'outputs': 'output1'},
            {'name': 'func1', 'func': lambda x: x, 'inputs': 'input2', 'outputs': 'output2'}
        ]
        expected_output = [
            [{
                'name': 'func1',
                'train': {
                    'name': 'func1',
                    'func': model_space[0]['func'],
                    'inputs': ['input1'],
                    'outputs': ['output1'],
                    'args': [],
                    'kwargs': {}
                },
                'inference': {
                    'name': 'func1',
                    'func': model_space[0]['func'],
                    'inputs': ['input1'],
                    'outputs': ['output1'],
                    'args': [],
                    'kwargs': {}
                }
            }],
            [{
                'name': 'func1_1',
                'train': {
                    'name': 'func1_1',
                    'func': model_space[1]['func'],
                    'inputs': ['input2'],
                    'outputs': ['output2'],
                    'args': [],
                    'kwargs': {}
                },
                'inference': {
                    'name': 'func1_1',
                    'func': model_space[1]['func'],
                    'inputs': ['input2'],
                    'outputs': ['output2'],
                    'args': [],
                    'kwargs': {}
                }
            }]
        ]
        result = validate_config(model_space)
        self.assertEqual(result, expected_output)

    def test_correct_input_not_modified(self):
        model_space = [
            [
                {
                    'name': 'test_func',
                    'train': {
                        'func': lambda x: x,
                        'inputs': ['input'],
                        'outputs': ['output'],
                        'args': [],
                        'kwargs': {}
                    },
                    'inference': {
                        'func': lambda x: x,
                        'inputs': ['input'],
                        'outputs': ['output'],
                        'args': [],
                        'kwargs': {}
                    }
                }
            ]
        ]
        model_space_copy = deepcopy(model_space)
        validate_config(model_space)
        self.assertEqual(model_space, model_space_copy)

    def test_validate_config_missing_keys(self):
        model_space = {
            'name': 'test_func'
            # Missing 'func', 'inputs', 'outputs'
        }
        with self.assertRaises(ValueError) as context:
            validate_config(model_space)
        self.assertIn("No 'train', 'inference' or 'func' key found.", str(context.exception))

    def test_validate_config_raises_error(self):
        model_space = [{'name': 'test_func'}]  # Missing 'func' key to trigger error in validate_function_dict
        with self.assertRaises(ValueError) as context:
            validate_config(model_space)
        assert "No 'train', 'inference' or 'func' key found." in str(context.exception)

    def test_validate_config_dict_without_name(self):
        model_space = [{'inputs': 'input', 'outputs': 'output'}]  # Missing 'name' key
        with self.assertRaises(ValueError) as context:
            validate_config(model_space)
        assert "No name provided and no function to pull name from" in str(context.exception)


if __name__ == '__main__':
    unittest.main()
