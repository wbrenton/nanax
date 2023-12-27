import numpy as np
from dataclasses import dataclass
from typing import Literal

@dataclass
class GridSearchSampler:
    """
    Given the list of 'spaces' it constructs a grid of all combinations.
    As trials are logged, the grid is updated to remove the trials that have already been evaluated.
    Each trial is a dictionary of the form {'array': np.array, 'dict': dict, 'score': float} where 'array' is a np.array 
    of the hyperparameters to try, 'dict' is a dictionary of the hyperparameters of format {'name', value} (used to contruct Args)
    and 'score' is the score of the trial, which is populated after the trial is evaluated.
    """
    spaces: list

    def __post_init__(self):
        space_samples = []
        for space in self.spaces:
            space_samples.append(space.values())
        self.grid = np.array(np.meshgrid(*space_samples)).T.reshape(-1, len(self.spaces))
        
    def sample(self, num_samples=1):
        # return num_samples^len(self.spaces) samples
        samples = []
        for space in self.spaces:
            samples.append(space.sample())

    def sample_all(self):
        return [{'array': configuration, 'args': self._todict(configuration)} for configuration in self.grid]

    def _todict(self, values_array):
        return {space.name: value.astype(space.dtype) for space, value in zip(self.spaces, values_array)}


class RandomSearchSampler:

    def __init__(self, spaces) -> None:
        self.spaces = spaces
        self.space_names = {space.name: space for space in self.spaces}

    def sample(self, num_samples):
        samples = []
        for space in self.spaces:
            sample = space.sample(num_samples)
            samples.append(sample[:, None])

        values_array = np.concatenate(samples, axis=1)
        return [
            {
                'args': (args := self._to_dict(value_array)),
                'args_str': self._args_str_id(args)
            }
            for value_array in values_array
        ]

    def update_center(self, center):
        for space in self.spaces:
            space.update_center(center)

    def to_array(self, args, normalize=False):
        array = []
        for name, value in args.items():
            if name in self.space_names:
                value = np.array(value).astype(self.space_names[name].dtype)
                if normalize:
                    value = self.space_names[name].normalize(value)
                array.append(value)
            else:
                raise ValueError(f"Hyperparameter {name} not found in sampler spaces.")
        return np.array(array)

    def _to_dict(self, values_array):
        return {space.name: value.astype(space.dtype) for space, value in zip(self.spaces, values_array)}
    
    def _args_str_id(self, args):
        return "-".join([f"{key}={value}" for key, value in args.items()])