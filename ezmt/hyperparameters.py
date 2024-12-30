import random
from abc import ABC, abstractmethod


class Hyperparameter(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def mutate(self, current_value):
        pass


class ContinuousRange(Hyperparameter):
    def __init__(self, start, end, max_change_perc=0.05):
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"Start and end values for ContinuousRange must be numeric. Got {start}, {end}")
        if start >= end:
            raise ValueError(f"Start value must be less than end value. Got start={start} and end={end}")
        self.start = start
        self.end = end
        self.max_change_perc = max_change_perc

    def sample(self):
        """Sample a random value from the continuous range."""
        return random.uniform(self.start, self.end)

    def mutate(self, current_value):
        new = current_value + (self.end - self.start) * random.uniform(-self.max_change_perc, self.max_change_perc)
        new = max(self.start, min(self.end, new))
        new = round(new, 5)
        return new


class DiscreteOrdinal(Hyperparameter):
    def __init__(self, options, max_index_change=1):
        if not hasattr(options, "__iter__") or isinstance(options, str):
            raise ValueError(f"Options for DiscreteChoice must be a non-string iterable. Got {type(options)}")
        self.options = options
        self.max_index_change = max_index_change

    def sample(self):
        """Sample a random value from the discrete choice options."""
        return random.choice(self.options)

    def mutate(self, current_value):
        index_diff = 0
        # avoid choosing same index multiple times in a row
        while index_diff == 0:
            index_diff = random.randint(-self.max_index_change, self.max_index_change)
        new_index = self.options.index(current_value) + index_diff
        new_index = max(0, min(len(self.options) - 1, new_index))
        return self.options[new_index]


class DiscreteChoice(Hyperparameter):
    def __init__(self, options):
        if not isinstance(options, (list, tuple)):
            raise ValueError(f"Options for DiscreteChoice must be a list or tuple. Got {type(options)}")
        self.options = options

    def sample(self):
        """Sample a random value from the discrete choice options."""
        return random.choice(self.options)

    def mutate(self, current_value):
        new = current_value
        while new == current_value:
            new = self.sample()
        return new
