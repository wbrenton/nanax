import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import os
import time
import numpy as np
from typing import List
from dataclasses import dataclass

from tune.slurm_tuner.tuner import Trial, Monitor, SlurmManager
from tune.slurm_tuner.spaces import NormalDiscreteSpace, NormalContinuousSpace, UniformDiscreteSpace
from tune.slurm_tuner.samplers import RandomSearchSampler

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

SLEEP = 1

@dataclass
class Config:
    dir: str
    envs: List[str]
    seeds: List[int]
    verbose: bool
    launch_script: str
    objective_metric: str
    include_env: bool = False
    init_samples: int = 5
    num_samples: int = 1000
    optimization_iters: int = 10
    is_restart: bool = False


class BayesOptStudy:

    def __init__(self, config: Config, sampler, pruner=None) -> None:
        self.config = config
        self.sampler = sampler
        self.pruner = pruner
        self.trials = []

    def start(self):
        self._proactive_asserts()
        self._start_manager()
        self._start_monitor()
        self.optimization_loop()

    def optimization_loop(self):
        self._setup()

        # evaulate initial samples
        if self.config.is_restart:
            self._update_training_data()
        else:
            hp_list = self._sample(init=True)
            self._execute_trials(hp_list)
        for iter in range(self.config.optimization_iters):
            # train gaussian process
            X, y = self._train_data()
            self.gp.fit(X, y)

            # sample next hyperparameters, via samlpler and acquisition function
            hp_list = self._sample()
            self._execute_trials(hp_list, iter+1)

        print(f"Best hyperparameters: {self.best_args} with {self.best_performance}.")
        self.stop()

    def _setup(self):
        self.best_args = None
        self.best_performance = -np.inf
        self.train_data = {}

        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    def _sample(self, init=False):
        # sample from search space
        if init:
            hp_list = self.sampler.sample(self.config.init_samples)
        else:
            hp_list = self.sampler.sample(self.config.num_samples)
            normalized = []
            for hp in hp_list:
                normalized.append(self.sampler.to_array(hp['args'], normalize=True))
            X_candidate = np.stack(normalized, axis=0)
            X, y = self._train_data()
            ei = self._expected_improvement(X_candidate, X)
            idx = np.argmax(ei)
            hp_list = [hp_list[idx]]

        return hp_list

    def _expected_improvement(self, X_cand, X_train):
        # Best observed value
        y_best = np.max(self.gp.predict(X_train))

        # Predict mean and std for candidates
        mu, sigma = self.gp.predict(X_cand, return_std=True)

        # Calculate the improvement
        with np.errstate(divide='warn'):
            imp = mu - y_best
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _train_data(self):
        Xs, ys = [], []
        for _, (X, y) in self.train_data.items():
            Xs.append(X)
            ys.append(y)
        X = np.stack(Xs, axis=0)
        y = np.stack(ys)
        return X, y

    def _execute_trials(self, hp_list, iter=0):
        for hp in hp_list:
            trial = Trial(
                args=hp['args'],
                config=self.config,
                callback=self.manager.add_job
            )
            trial.start()
            self.trials.append(trial)

        # wait for trials to finish
        while len(self.trials) > 0:
                time.sleep(SLEEP)

        # update training data with new results
        self._update_training_data()

        # move the center to the best performing hyperparameters
        self._update_search_centers(iter)

    def _update_training_data(self):
        results = self.monitor.load_results()
        for args_str, metric in results.items():
            if args_str not in self.train_data:
                print(f"{args_str} {100*metric[-1]:.4f}")
                args = self.convert_args_str_to_args(args_str)
                X_normalized = self.sampler.to_array(args, normalize=True)
                self.train_data[args_str] = (X_normalized, metric[-1])
    
    def _update_search_centers(self, iter):
        args, performance = self.monitor.get_best_args(self.best_performance, iter)
        if performance > self.best_performance:
            print(f"{iter}: {args} {performance} improved from {self.best_performance}")
            self.best_args = args
            self.best_performance = performance
            self._update_sampler(self.best_args)

    def _update_sampler(self, args):
        args = self.convert_args_str_to_args(args)
        space_names = {space.name: space for space in self.sampler.spaces}
        for name, value in args.items():
            if name in space_names:
                space = space_names[name]
                if space.dtype == float:
                    float(value)
                elif space.dtype == int:
                    int(value)
                space.update_center(value)
            else:
                raise ValueError(f"Hyperparameter {name} not found in sampler spaces.")
            
    def convert_args_str_to_args(self, args_str):
        hps = args_str.split("-")
        args = {}
        for hp in hps:
            name, value = hp.split("=")
            args[name] = value
        return args

    def _start_manager(self):
        self.manager = SlurmManager()
        self.manager.start()

    def _start_monitor(self):
        self.monitor = Monitor(self.config.dir, self.trials self.config.verbose)
        self.monitor.start()

    def stop(self):
        self.monitor.stop()
        self.manager.stop()

    def _proactive_asserts(self):
        # assert that the launch script exists
        launch_script = self.config.launch_script
        assert os.path.isfile(launch_script), f"Launch script {launch_script} does not exist."


if __name__ == "__main__":
    lr_space = NormalContinuousSpace(name='learning_rate', dtype=float, center=0.01, radius=0.25, min=1e-4, max=0.1, space_type='linear')
    batch_size_space = UniformDiscreteSpace(name='batch_size', dtype=int, values=[2**i for i in range(3, 11)], space_type='linear')

    config = Config(
        dir="runs",
        envs=["mnist"],
        seeds=[1],
        launch_script=sys.path[-1] + "tune/examples/script.sh",
        objective_metric="test/accuracy",
        num_samples=3
        verbose=False
    )

    expirement = BayesOptStudy(
        config=config,
        sampler=RandomSearchSampler([lr_space, batch_size_space]),
    )
    expirement.start()