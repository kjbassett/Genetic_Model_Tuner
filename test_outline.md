## Initialization and Configuration Tests:

* ~~Test Initialization with Valid Configuration: Ensure ModelTuner initializes properly with valid model space, data, and other parameters.~~
* ~~Test Initialization with Invalid Model Space: Confirm that ModelTuner raises an error if model_space is invalid or fails validation.~~
* ~~Test Setting Goals (min/max): Check that goal parameter correctly impacts fitness scoring logic, setting up the tuner for either minimization or maximization.~~
* ~~Test Train Input Availability: Check that each input in the train side of the pipeline is either part of the dataset or an output from a previous train step~~
* Test Inference Input Availability: Check that each input in the inference side of the pipeline is either part of the dataset, an output from a previous train step, or an output from a previous inference step.
* Test Function Availability: Check that if the value of a 'func' key is a string, the string is an outputs from previous steps. Strings with notation should for only the first word. (model.run => model is an output) 

## Population Initialization Tests:

* ~~Test Initial Population Creation: Verify that populate_init creates a population of the correct size and structure based on population_size and model space.~~
* ~~Test Initial Population Diversity: Ensure that the initial population has varied genes based on random selection from model_space, confirming non-uniformity in generated organisms.~~

## Selection and Reproduction Tests:

* ~~Test Asexual Reproduction: Validate that select_and_reproduce with asexual reproduction produces a new generation, with children that inherit genes from one parent.~~
* Test Sexual Reproduction: Test sexual reproduction by verifying that offspring inherit genes from two different parents.
* ~~Test Elitism Preservation: Ensure that the top organisms from the previous generation (based on elitism setting) are preserved in the new generation.~~
* ~~Test Mutation Effects: Confirm that mutations introduce variability, especially in continuous nucleotides, and respect func_prob, nuc_prob, and mutation limits.~~

## Experience Population Tests:

* ~~Test that gpu steps are executed by the parents process~~
* ~~Test the cpu steps are executed in the processing pool~~
* ~~For any given step i, test that members of the population with gene i having gpu=True run after all the members with gpu=False genes start their processes~~
* ~~Test that two members of the population with the same dna up to gene i don't make_decision twice during step i.~~
* ~~Test that make_decision and pool.apply_async were called the right number of times~~

## Decision-Making Tests:

* Test Valid Decision Execution: Verify that make_decision correctly retrieves and executes a function based on the current gene, with outputs correctly assigned.
* Test Decision with Invalid Function Reference: Ensure that if the method make_decision is given an invalid function reference, it raises an appropriate exception.
* Test Argument Passing and Output Collection: Confirm that make_decision passes arguments and keyword arguments as specified in each gene, and the output updates state accurately.

## Fitness Scoring Tests:

* Test Fitness Calculation for Minimization: Simulate scores and verify that fitness scores are calculated accurately, with the correct organism marked as “best” when the goal is min.
* Test Fitness Calculation for Maximization: Similar to the minimization test, but with a goal of max, ensuring that the highest score is considered the best.
* Test Population Metrics: Confirm that average, variance, best, and worst metrics are correctly calculated and stored in metrics.

## End-to-End Run Tests:

* Test Single Generation Run: Execute a single generation with both population initialization and fitness scoring to verify end-to-end functionality.
* Test Multi-Generation Run: Run multiple generations and verify that the tuner progresses through generations with expected evolutionary changes in the population.
* Test Performance with GPU and Non-GPU Steps: Run a test with both GPU and non-GPU steps to ensure the workflow integrates them seamlessly, with proper inter-process handling and no task loss.
* Test Run Output Consistency: Ensure that run returns the highest-performing organism as expected, confirming end-to-end optimization consistency.

## Additional Tests
* Test Error Handling in Initialization: Confirm error handling for various incorrect configurations, such as missing data columns or invalid generational settings.
* Test Organism Mutation Variability: Confirm that mutations adjust continuously within the set limits, particularly for continuous nucleotides, and evaluate mutation magnitude over runs.

## Organism Tests
* Test that an organism can be saved
* Test that an organism can be loaded for inference
* Test that the inference version of the organism has the same performance as the train version 
* Test that the correct parameters and objects are chosen to save