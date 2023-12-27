from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal, List

import numpy as np
from scipy.stats import norm

"""
Continuous Space:
    Uniform Distribution:
        Linear Space
        Log Space
        Logit Space
    Normal Distribution:
        Linear Space
        Log Space
        Logit Space
Discrete Space:
    Uniform Distribution:
        Linear Space
        Log Space
        Logit Space
    Normal Distribution:
        Linear Space
        Log Space
        Logit Space
"""

@dataclass
class Space(ABC):
    name: str
    dtype: type
    space_type: Literal['linear', 'log', 'logit']

    center: float = None
    radius: float = None
    min: float = None
    max: float = None
    values: List[float] = field(default_factory=list)

    def __post_init__(self):
        assert self.space_type in ['linear', 'log', 'logit'], f"Space type {self.space_type} not supported."
        assert self.dtype in [int, float], f"Data type {self.dtype} not supported."

    def sample(self, num_samples):
        return self._safe_sample(num_samples).astype(self.dtype)

    def _safe_sample(self, num_samples):
        samples = np.array([])
        while len(samples) < num_samples:
            samples = np.concatenate([samples, self._sample(num_samples - len(samples))])
            if self.min is not None:
                samples = samples[samples > self.min]
            if self.max is not None:
                samples = samples[samples < self.max]
        return samples

    def normalize(self, value):
        return (value - self.min) / (self.max - self.min)

    @abstractmethod
    def _sample(self, center):
        pass

    @abstractmethod
    def update_center(self, center):
        pass


##########################################
############ Continuous Space ############
##########################################


@dataclass
class ContinuousSpace(Space):

    def __post_init__(self):
        assert self.min < self.max, f"Min {self.min} must be less than max {self.max}."

    def values(self):
        if self.space_type == 'linear':
            return np.linspace(self.min, self.max, num_samples)
        elif self.space_type == 'log':
            return np.logspace(self.min, self.max, num_samples)
        elif self.space_type == 'logit':
            return np.logit(np.linspace(self.min, self.max, num_samples))


class UniformContinuousSpace(ContinuousSpace):

    def update_center(self, center):
        return

    def _sample(self, num_samples):
        if self.space_type == 'linear':
            return np.random.uniform(self.min, self.max, num_samples)
        elif self.space_type == 'log':
            min_exp, max_exp = np.log10(self.min), np.log10(self.max)
            log_space_values = np.logspace(min_exp, max_exp, num=int(1e6))
            return np.random.choice(log_space_values, num_samples, replace=False)
        elif self.space_type == 'logit':
            logit_min = np.log(p_min / (1 - p_min))
            logit_max = np.log(p_max / (1 - p_max))
            samples_logit_space = np.random.uniform(logit_min, logit_max, num_samples)
            return 1 / (1 + np.exp(-samples_logit_space))


class NormalContinuousSpace(ContinuousSpace):

    def __post_init__(self):
        assert self.center >= self.min and self.center <= self.max, f"Center {self.center} must be between min {self.min} and max {self.max}."
        assert self.radius >= 0 and self.radius <= 1, f"Radius {self.radius} must be between 0 and 1. It's applied as a percentage of the center."
    
    def update_center(self, center):
        self.center = center

    def _sample(self, num_samples):
        if self.space_type == 'linear':
            return np.random.normal(self.center, self.sigma, num_samples)

        elif self.space_type == 'log':
            min_exp, max_exp = np.log10(self.min), np.log10(self.max)
            log_space_values = np.logspace(min_exp, max_exp, num=int(1e6))
            gaussian_values = norm.pdf(log_space_values, self.center, self.sigma)
            probs = gaussian_values / gaussian_values.sum()
            return np.random.choice(log_space_values, num_samples, replace=False, p=probs)

        elif self.space_type == 'logit':
            logit_center = np.log(self.center / (1 - self.center))
            logit_sigma = np.log(self.sigma / (1 - self.sigma))
            samples_logit_space = np.random.normal(logit_center, logit_sigma, num_samples)
            return 1 / (1 + np.exp(-samples_logit_space))

    @property
    def sigma(self):
        return self.radius * self.center


##########################################
############ Discrete Space ############
##########################################


@dataclass
class DiscreteSpace(Space):

    def __post_init__(self):
        self.values = np.array(self.values).astype(self.dtype)
        self.min = self.values.min()
        self.max = self.values.max()

class UniformDiscreteSpace(DiscreteSpace):

    def update_center(self, center):
        return

    def _sample(self, num_samples):
        return np.random.choice(self.values, num_samples)

class NormalDiscreteSpace(DiscreteSpace):
    """
    'center' corresponds to the bin that the gaussian is centered on
    'radius' corresponds to the % of the total bins that the gaussian spans (i.e. the standard deviation around the center)
    """

    def update_center(self, center):
        self.center = center # this is the index of the center value in self.values
        assert self.center in self.values, f"Center {self.center} must be in values {self.values}."

    def _sample(self, num_samples):
        if self.space_type == 'linear':
            loc = np.where(self.values == self.center)[0][0]
            gaussian_values = norm.pdf(range(len(self.values)), loc=loc, scale=self.sigma)
            probs = gaussian_values / gaussian_values.sum()
            # TODO :remove
            # plot the gaussian values
            plt.plot(range(len(self.values)), probs)
            plt.savefig('probs_values.png')
            # probs = gaussian_values / gaussian_values.sum()
            # apply softmax
            return np.random.choice(self.values, num_samples, replace=True, p=probs)

    @property
    def sigma(self):
        return self.radius * len(self.values)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test = "discrete" # "continuous"

    if test == "continuous":
        linear_uniform_continuous_space = UniformContinuousSpace(name='learning_rate', dtype=float, min=1e-4, max=0.1, space_type='linear')
        log_uniform_continuous_space = UniformContinuousSpace(name='learning_rate', dtype=float, min=1e-4, max=0.1, space_type='log')
        linear_normal_continuous_space = NormalContinuousSpace(name='learning_rate', dtype=float, center=0.01, radius=0.25, min=1e-4, max=0.1, space_type='linear')
        log_normal_continuous_space = NormalContinuousSpace(name='learning_rate', dtype=float, center=0.01, radius=0.25, min=1e-4, max=0.1, space_type='log')
        linear_normal_continuous_space.sample(1000)
        spaces = [
            linear_uniform_continuous_space,
            log_uniform_continuous_space,
            linear_normal_continuous_space,
            log_normal_continuous_space
        ]
        def draw_samples(space, num_samples=100):
            return space.sample(num_samples)

        titles = ['Linear Uniform', 'Log Uniform', 'Linear Normal', 'Log Normal']
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        for ax, space, title in zip(axs.ravel(), spaces, titles):
            samples = draw_samples(space)
            ax.hist(samples, bins=50, color='blue', alpha=0.7)
            ax.set_title(title)
            ax.grid(True)

        plt.tight_layout()
        plt.show()
        plt.savefig(f'{test}_spaces.png')

    if test == "discrete":
        lr_values = [2**i for i in range(3, 11)] #[1e-5, 1.25e-5, 1.5e-5, 1.75e-5, 1e-4, 1.25e-4, 1.5e-4, 1.75e-4, 1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 1e-2, 1.25e-2, 1.5e-2, 1.75e-2, 1e-1]
        linear_uniform_discrete_space = UniformDiscreteSpace(name='learning_rate', dtype=float, values=lr_values, space_type='linear')
        linear_normal_discrete_space = NormalDiscreteSpace(name='learning_rate', dtype=float, center=64, radius=0.05, values=lr_values, space_type='linear')
        spaces = [
            linear_uniform_discrete_space,
            linear_normal_discrete_space,
        ]
        def draw_samples(space, num_samples=1000):
            return space.sample(num_samples)

        titles = ['Uniform', 'Normal']
        fig, axs = plt.subplots(2, figsize=(10, 8))

        for ax, space, title in zip(axs.ravel(), spaces, titles):
            samples = draw_samples(space)
            ax.hist(samples, bins=100, color='blue', alpha=0.7)
            ax.set_title(title)
            ax.grid(True)

        plt.tight_layout()
        plt.show()
        plt.savefig(f'{test}_spaces.png')