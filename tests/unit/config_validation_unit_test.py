import unittest
from ezmt.config_validation import validate_config
from ezmt.hyperparameters import ContinuousRange
from copy import deepcopy


class TestValidateConfig(unittest.TestCase):

    def test_validate_config_empty_list(self):
        model_space = []
        hyperparams = {}
        expected_output = []
        self.assertEqual(validate_config(model_space, hyperparams), expected_output)

    def test_single_func_standardized_to_train_inference(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'args': 'x_train',
            'outputs': 'output'
        }

        result = validate_config(model_space, {})
        assert 'train' in result[0][0]
        assert 'inference' in result[0][0]
        assert result[0][0]['train']['func'] == result[0][0]['inference']['func'] == model_space['func']

    def test_valid_inputs_and_outputs(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'args': 'x_train',  # Single string
            'outputs': ['predictions']  # List of strings
        }

        result = validate_config(model_space, {})
        assert result[0][0]['train']['args'] == ['x_train']
        assert result[0][0]['train']['outputs'] == ['predictions']

    def test_multiple_funcs_standardized_to_train_inference(self):
        model_space = [
            {'name': 'test_func1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
            {'name': 'test_func2', 'func': lambda y: y, 'args': 'x_train', 'outputs': 'output2'}
        ]

        result = validate_config(model_space, {})
        assert result[0][0]['train']['func'] == result[0][0]['inference']['func']
        assert result[1][0]['train']['func'] == result[1][0]['inference']['func']

    def test_multiple_funcs_input_output_standardization(self):
        model_space = [
            {'name': 'test_func1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
            {'name': 'test_func2', 'func': lambda y: y, 'args': 'x_train', 'outputs': 'output2'}
        ]

        result = validate_config(model_space, {})
        assert result[0][0]['train']['args'] == ['x_train']
        assert result[0][0]['train']['outputs'] == ['output1']
        assert result[1][0]['train']['args'] == ['x_train']
        assert result[1][0]['train']['outputs'] == ['output2']

    def test_pipeline_step_with_multiple_function_choices(self):
        model_space = [
            [
                {'name': 'step1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
                {'name': 'step2', 'func': lambda x: x, 'args': ['x_train'], 'outputs': ['output2']}
            ],
            [
                {'name': 'step3', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output3'}
            ]
        ]

        result = validate_config(model_space, {})
        assert result[0][0]['name'] == 'step1'
        assert result[0][1]['name'] == 'step2'
        assert result[1][0]['name'] == 'step3'

    def test_raises_error_on_too_deep_nesting(self):
        model_space = [
            [
                [
                    {'name': 'func1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
                    {'name': 'func2', 'func': lambda y: y, 'args': 'x_train', 'outputs': 'output2'}
                ],
                [
                    {'name': 'func3', 'func': lambda z: z, 'args': 'x_train', 'outputs': 'output3'}
                ]
            ],
            [
                {'name': 'func4', 'func': lambda a: a, 'args': 'x_train', 'outputs': 'output4'}
            ]
        ]
        with self.assertRaises(ValueError) as context:
            validate_config(model_space, {})
        assert "The third layer in should be a dictionary. Got <class 'list'>" in str(context.exception)

    def test_mixed_dict_and_list_handling_in_model_space(self):
        model_space = [
            {'name': 'func1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
            [
                {'name': 'func2', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output2'},
                {'name': 'func3', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output3'}
            ]
        ]

        result = validate_config(model_space, {})
        assert result[0][0]['name'] == 'func1'
        assert result[1][0]['name'] == 'func2'
        assert result[1][1]['name'] == 'func3'

    def test_name_collision_resolves_with_unique_suffix(self):
        model_space = [
            {'name': 'func1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
            {'name': 'func1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output2'}
        ]

        result = validate_config(model_space, {})
        assert result[0][0]['name'] == 'func1'
        assert result[1][0]['name'] == 'func1_1'

    def test_input_not_modified_by_validation(self):
        model_space = [
            [
                {
                    'name': 'test_func',
                    'train': {
                        'func': lambda x: x,
                        'args': ['x_train'],
                        'kwargs': {},
                        'outputs': ['output'],
                        'gpu': False
                    },
                    'inference': {
                        'func': lambda x: x,
                        'args': ['x_train'],
                        'kwargs': {},
                        'outputs': ['output'],
                        'gpu': False
                    }
                }
            ]
        ]
        model_space_copy = deepcopy(model_space)
        validate_config(model_space, {})
        self.assertEqual(model_space, model_space_copy)

    def test_missing_func_raises_value_error(self):
        model_space = {
            'name': 'test_func'
            # Missing 'func', 'args', 'outputs'
        }
        with self.assertRaises(ValueError) as context:
            validate_config(model_space, {})
        self.assertIn("No 'train', 'inference' or 'func' key found.", str(context.exception))

    def test_missing_name_raises_value_error(self):
        model_space = [{'args': 'input', 'outputs': 'output'}]  # Missing 'name' key
        with self.assertRaises(ValueError) as context:
            validate_config(model_space, {})
        assert "No name provided and no function to pull name from" in str(context.exception)

    def test_missing_inputs_defaults_to_empty_list(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'outputs': 'output'  # Missing 'args'
        }

        result = validate_config(model_space, {})
        assert result[0][0]['train']['args'] == []
        assert result[0][0]['inference']['args'] == []

    def test_missing_outputs_defaults_to_empty_list(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'args': 'x_train'  # Missing 'outputs'
        }

        result = validate_config(model_space, {})
        assert result[0][0]['train']['outputs'] == []
        assert result[0][0]['inference']['outputs'] == []

    def test_missing_kwargs_defaults_to_empty_dict(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'args': 'x_train',  # Missing 'kwargs'
            'outputs': 'output'
        }

        result = validate_config(model_space, {})
        assert result[0][0]['train']['kwargs'] == {}
        assert result[0][0]['inference']['kwargs'] == {}

    def test_valid_raw_arg_not_in_list(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'args': 123,  # Invalid non-iterable input
        }

        validate_config(model_space, {})

    def test_predefined_inputs_are_accepted(self):
        model_space = [
            {'name': 'step1', 'func': lambda x: x + 1, 'args': 'x_train', 'outputs': 'step1_output'},
            {'name': 'step2', 'func': lambda x: x * 2, 'args': 'x_test', 'outputs': 'step2_output'},
            {'name': 'step3', 'func': lambda y: y * 2, 'args': 'y_train', 'outputs': 'step3_output'},
            {'name': 'step4', 'func': lambda z: z * 3, 'args': 'y_test', 'outputs': 'step4_output'}
        ]

        result = validate_config(model_space, {})
        assert result[0][0]['train']['args'] == ['x_train']
        assert result[1][0]['train']['args'] == ['x_test']
        assert result[2][0]['train']['args'] == ['y_train']
        assert result[3][0]['train']['args'] == ['y_test']

    def test_outputs_of_previous_steps_are_accepted_as_inputs(self):
        model_space = [
            {'name': 'step1', 'func': lambda x: x + 1, 'args': 'x_train', 'outputs': 'step1_output'},
            {'name': 'step2', 'func': lambda x: x * 2, 'args': 'step1_output', 'outputs': 'step2_output'},
            {'name': 'step3', 'func': lambda y: y * 2, 'args': 'step2_output', 'outputs': 'step3_output'}
        ]

        result = validate_config(model_space, {})
        assert result[1][0]['train']['args'] == ['step1_output']
        assert result[2][0]['train']['args'] == ['step2_output']

    def test_raw_string_arg_raises_no_error(self):
        model_space = [
            {'name': 'step1', 'func': lambda x: x + 1, 'args': 'x_train', 'outputs': 'step1_output'},
            {'name': 'step2', 'func': lambda x: x * 2, 'args': 'raw_string_arg', 'outputs': 'step2_output'}
        ]

        validate_config(model_space, {})

    def test_invalid_non_iterable_outputs_raises_error(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'outputs': 456  # Invalid non-iterable output
        }

        with self.assertRaises(TypeError) as context:
            validate_config(model_space, {})
        assert "'outputs' must be a string or a list" in str(context.exception)

    def test_valid_args_with_custom_type(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'args': [ContinuousRange(0, 10), [1, 2, 3]]
        }

        result = validate_config(model_space, {})
        assert isinstance(result[0][0]['train']['args'][0], ContinuousRange)
        assert result[0][0]['train']['args'][1] == [1, 2, 3]

    def test_valid_raw_arg_raises_no_error(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'args': [123]  # Invalid: not a ContinuousRange or iterable
        }

        validate_config(model_space, {})


    def test_valid_kwargs_with_custom_type(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'kwargs': {'param1': ContinuousRange(0, 10), 'param2': [1, 2, 3]}
            # Valid ContinuousRange and discrete options
        }

        result = validate_config(model_space, {})
        assert isinstance(result[0][0]['train']['kwargs']['param1'], ContinuousRange)
        assert result[0][0]['train']['kwargs']['param2'] == [1, 2, 3]

    def test_valid_raw_kwargs_raise_no_error(self):
        model_space = {
            'name': 'test_func',
            'func': lambda x: x,
            'kwargs': {"x": 123}  # Invalid: not a ContinuousRange or iterable
        }

        validate_config(model_space, {})

    def test_func_as_reference_to_previous_step_output(self):
        model_space = [
            {'name': 'step1', 'func': lambda x: x + 1, 'args': 'x_train', 'outputs': 'step1_output'},
            {'name': 'step2', 'func': 'step1_output', 'args': 'step1_output', 'outputs': 'step2_output'}
        ]

        result = validate_config(model_space, {})
        assert 'train' in result[1][0]
        assert result[1][0]['train']['func'] == 'step1_output'
        assert result[1][0]['train']['args'] == ['step1_output']

    def test_func_reference_valid_with_multiple_previous_outputs(self):
        model_space = [
            {'name': 'step1', 'func': lambda x: x + 1, 'args': 'x_train',
             'outputs': ['step1_output1', 'step1_output2']},
            {'name': 'step2', 'func': 'step1_output2', 'args': 'step1_output2', 'outputs': 'step2_output'}
        ]

        result = validate_config(model_space, {})
        assert 'train' in result[1][0]
        assert result[1][0]['train']['func'] == 'step1_output2'
        assert result[1][0]['train']['args'] == ['step1_output2']

    def test_func_reference_raises_error_if_no_match_in_previous_outputs(self):
        model_space = [
            {'name': 'step1', 'func': lambda x: x + 1, 'args': 'x_train',
             'outputs': ['step1_output1', 'step1_output2']},
            {'name': 'step2', 'func': 'non_existent_output', 'args': 'step1_output1', 'outputs': 'step2_output'}
        ]

        with self.assertRaises(ValueError) as context:
            validate_config(model_space, {})
        assert "Function reference 'non_existent_output' not found in any previous outputs" in str(context.exception)

    def test_valid_configuration_with_func_as_callable_or_reference(self):
        model_space = [
            {'name': 'step1', 'func': lambda x: x + 1, 'args': 'x_train', 'outputs': 'step1_output'},
            {'name': 'step2', 'func': 'step1_output', 'args': 'step1_output', 'outputs': 'step2_output'},
            {'name': 'step3', 'func': lambda y: y * 2, 'args': 'step2_output', 'outputs': 'step3_output'}
        ]

        result = validate_config(model_space, {})
        assert result[1][0]['train']['func'] == 'step1_output'
        assert result[2][0]['train']['func'].__name__ == (lambda y: y * 2).__name__

    def test_single_level_nested_structure(self):
        model_space = [
            [
                {'name': 'step1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
                {'name': 'step2', 'func': lambda x: x * 2, 'args': 'x_train', 'outputs': 'output2'}
            ],
            [
                {'name': 'step3', 'func': lambda x: x + 1, 'args': 'x_train', 'outputs': 'output3'}
            ]
        ]

        result = validate_config(model_space, {})
        assert result[0][0]['train']['args'] == ['x_train']
        assert result[0][0]['train']['outputs'] == ['output1']
        assert result[1][0]['train']['args'] == ['x_train']

    def test_raises_error_on_too_deep_nesting_corrected(self):
        model_space = [
            [
                [
                    {'name': 'func1', 'func': lambda x: x, 'args': 'x_train', 'outputs': 'output1'},
                ]
            ]
        ]

        with self.assertRaises(ValueError) as context:
            validate_config(model_space, {})
        assert "The third layer in should be a dictionary. Got <class 'list'>" in str(context.exception)

    def test_continuous_range_valid(self):
        range1 = ContinuousRange(0, 10)
        assert range1.start == 0
        assert range1.end == 10

    def test_continuous_range_sample(self):
        range1 = ContinuousRange(0, 10)
        sample = range1.sample()
        assert 0 <= sample <= 10

    def test_invalid_continuous_range_raises_error(self):
        with self.assertRaises(ValueError):
            ContinuousRange(10, 5)  # Invalid: start > end
        with self.assertRaises(ValueError):
            ContinuousRange('a', 'b')  # Invalid: non-numeric values


if __name__ == '__main__':
    unittest.main()
