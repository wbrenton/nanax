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
from tune.slurm_tuner.spaces import ContinuousUniformSpace, ContinuousNormalSpace, UniformCategoricalSpace
from tune.slurm_tuner.samplers import RandomSearchSampler

SLEEP = 1

@dataclass
class Config:
    dir: str
    envs: List[str]
    seeds: List[int]
    launch_script: str
    objective_metric: str
    include_env: bool = False
    num_samples: int = 3
    optimization_iters: int = 10


class RandomSearchStudy:

    def __init__(self, config: Config, sampler, pruner=None) -> None:
        self.config = config
        self.sampler = sampler
        self.pruner = pruner
        self.trials = []

    def start(self):
        self._proactive_asserts()
        self._start_manager()
        self._start_monitor()
        best_args = None
        best_performance = -np.inf
        for iter in range(self.config.optimization_iters):
            hp_list = self.sampler.sample(self.config.num_samples)
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

            # determine best trial and update sampler
            args, performance = self.monitor.get_best_args(best_performance, iter)
            if performance > best_performance:
                best_args = args
                best_performance = performance
                self._update_sampler(best_args)

    def _update_sampler(self, args):
        # parse args str into a dictionary
        hps = args.split("-")
        space_names = {space.name: space for space in self.sampler.spaces}
        for hp in hps:
            name, value = hp.split("=")
            if name in space_names:
                space = space_names[name]
                if space.dtype == float:
                    float(value)
                elif space.dtype == int:
                    int(value)
                space.update_center(value)
            else:
                raise ValueError(f"Hyperparameter {name} not found in sampler spaces.")

    def _start_manager(self):
        self.manager = SlurmManager()
        self.manager.start()

    def _start_monitor(self):
        self.monitor = Monitor(self.config.dir, self.trials)
        self.monitor.start()

    def stop(self):
        self.monitor.stop()
        self.manager.stop()

    def _proactive_asserts(self):
        # assert that the launch script exists
        launch_script = self.config.launch_script
        assert os.path.isfile(launch_script), f"Launch script {launch_script} does not exist."


if __name__ == "__main__":
    lr_space = ContinuousUniformSpace(name='learning_rate', center=0.01, radius=0.1, min=1e-4, max=0.1, space_type='log')
    batch_size_space = UniformCategoricalSpace(name='batch_size', categories=np.array([32, 64, 128, 256, 512, 1024]), dtype=int)
    # batch_size_space = ContinuousUniformSpace(name='batch_size', center=64, radius=0.1, min=1, max=1000, space_type='linear', dtype=int)
    # in reality youd want to use a categorical space for batch size

    config = Config(
        dir="runs",
        envs=["mnist"],
        seeds=[1],
        launch_script=sys.path[-1] + "tune/examples/script.sh",
        objective_metric="test/accuracy",
        num_samples=3
    )

    expirement = RandomSearchStudy(
        config=config,
        sampler=RandomSearchSampler([lr_space, batch_size_space]),
    )
    expirement.start()