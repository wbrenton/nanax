import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm

@dataclass
class UniformCategoricalSpace:
    name: str
    categories: list
    dtype: type

    def sample(self, num_samples):
        num_samples = min(num_samples, len(self.categories))
        samples = np.random.choice(self.categories, num_samples, replace=False)
        return samples

    def values(self):
        return self.categories

    def update_center(self, center):
        pass
    
    def normalize(self, value):
        return (value - self.min) / (self.max - self.min)

    @property
    def max(self):
        return max(self.categories)

    @property
    def min(self):
        return min(self.categories)

@dataclass
class GaussianCategoricalSpace:
    name: str
    categories: list
    dtype: type
    center_idx: Optional[int] = None

    def __post_init__(self):
        if self.center_idx is None:
            self.center_idx = len(self.categories) // 2
            print(f"Center not specified for {self.name}, setting to {self.center_idx}")

    def sample(self, num_samples):
        # Create a Gaussian distribution centered around 'center'
        gaussian_probs = norm.pdf(range(len(self.categories)), loc=self.center_idx, scale=1)
        gaussian_probs /= gaussian_probs.sum()  # Normalize to sum to 1

        # Sample from the categories using the Gaussian probabilities
        samples = np.random.choice(self.categories, num_samples, replace=False, p=gaussian_probs)
        return samples

    def values(self):
        return self.categories


class ContinuousUniformSpace:

    def __init__(self, name, center, radius, min, max, space_type=Literal['linear', 'log', 'logit'], dtype=float) -> None:
        self.name = name
        self.center, self.radius = center, radius
        self.min, self.max = min, max
        self.space_type = space_type
        self.dtype = dtype

    def sample(self, num_samples):
        samples = self._safe_sample(num_samples).astype(self.dtype)
        return samples

    def update_center(self, center):
        self.center = center

    def _safe_sample(self, num_samples):
        samples = self._sample(num_samples)
        samples = np.clip(samples, self.min, self.max)

        idx = np.where(np.logical_or(samples == self.min, samples == self.max).any())
        samples = np.delete(samples, idx)
        while len(samples) != num_samples:
            new_samples = self._sample(len(idx))
            new_samples = np.clip(new_samples, self.min, self.max)
            samples = np.concatenate([samples, new_samples])
            idx = np.where(np.logical_or(samples == self.min, samples == self.max))
            samples = np.delete(samples, idx)
        return samples


    def _sample(self, num_samples):
        if self.space_type == 'linear':
            range = self.max - self.min
            min, max = self.center - self.radius * range, self.center + self.radius * range
            return np.random.uniform(min, max, num_samples)
        elif self.space_type == 'log':
            log_min, log_max = np.log(self.min + 1e-6), np.log(self.max + 1e-6)
            log_range = log_max - log_min
            min, max = log_min - self.radius * log_range, log_max + self.radius * log_range
            uniform_samples = np.random.uniform(log_min, log_max, num_samples)
            return np.exp(uniform_samples)
        elif self.space_type == 'logit':
            return np.random.logit(self.min, self.max, num_samples)


    def values(self, num_samples=100):
        if self.space_type == 'linear':
            return np.linspace(self.min, self.max, num_samples)
        elif self.space_type == 'log':
            return np.logspace(self.min, self.max, num_samples)
        elif self.space_type == 'logit':
            return np.logit(np.linspace(self.min, self.max, num_samples))

    def normalize(self, value):
        return (value - self.min) / (self.max - self.min)


class ContinuousNormalSpace:

    def __init__(self, name, center, radius, min, max, space_type=Literal['linear', 'log', 'logit'], dtype=float) -> None:
        self.name = name
        self.center, self.radius = center, radius
        self.min, self.max = min, max
        self.space_type = space_type
        self.dtype = dtype

    def sample(self, num_samples):
        samples = self._safe_sample(num_samples).astype(self.dtype)
        return samples

    def normalize(self, value):
        return (value - self.min) / (self.max - self.min)

    def update_center(self, center):
        self.center = center

    def _safe_sample(self, num_samples):
        samples = self._sample(num_samples)
        samples = np.clip(samples, self.min, self.max)
        
        idx = np.where(np.logical_or(samples == self.min, samples == self.max))
        samples = np.delete(samples, idx)
        while len(samples) != num_samples:
            new_samples = self._sample(len(idx))
            new_samples = np.clip(new_samples, self.min, self.max)
            samples = np.concatenate([samples, new_samples])
            idx = np.where(np.logical_or(samples == self.min, samples == self.max))
            samples = np.delete(samples, idx)
        return samples


    def _sample(self, num_samples):
        if self.space_type == 'linear':
            return np.random.normal(self.center, self.radius, num_samples)
        elif self.space_type == 'log':
            return np.random.lognormal(self.center, self.radius, num_samples)
        elif self.space_type == 'logit':
            return np.random.logit(self.center, self.radius, num_samples) # TODO: I don't think this is correct


    def values(self, num_samples=100):
        if self.space_type == 'linear':
            return np.linspace(self.min, self.max, num_samples)
        elif self.space_type == 'log':
            return np.logspace(self.min, self.max, num_samples)
        elif self.space_type == 'logit':
            return np.logit(np.linspace(self.min, self.max, num_samples))


if __name__ == "__main__":
    batch_sizes = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    
    batch_size_space = UniformCategoricalSpace(
        "batch_size",
        batch_sizes,
        dtype=int,
    )

    print("Uniform sample:")
    print(batch_size_space.sample(5))

    batch_size_space = GaussianCategoricalSpace(
        "batch_size",
        batch_sizes,
        dtype=int
    )

    print("Gaussian sample:")
    print(batch_size_space.sample(5))
