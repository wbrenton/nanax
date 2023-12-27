import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import os
import numpy as np
from typing import List

from tune.slurm_tuner.tuner import Config, Study, Trial, Monitor, SlurmManager
from tune.slurm_tuner.spaces import UniformCategoricalSpace
from tune.slurm_tuner.samplers import GridSearchSampler

class Config:
    dir: str
    envs: List[str]
    seeds: List[int]
    launch_script: str
    objective_metric: str
    include_env: bool = False


class Study:

    def __init__(self, config: Config, sampler, pruner=None) -> None:
        self.config = config
        self.sampler = sampler
        self.pruner = pruner
        self.trials = []

    def start(self):
        self._proactive_asserts()
        self._start_manager()
        self._start_monitor()
        hp_list = self.sampler.sample_all()
        for hp in hp_list:
            trial = Trial(
                args=hp['args'],
                config=self.config,
                callback=self.manager.add_job
            )
            trial.start()
            self.trials.append(trial)

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


    lr_space = UniformCategoricalSpace(
        "learning_rate",
        np.array([0.1]),#, 0.01, 0.001]),#linear_log_arange(1),
        dtype=float
    )

    batch_size_space = UniformCategoricalSpace(
        "batch_size",
        np.array([32, 128]),#, 64, 128]), #np.array([32, 64, 128, 256]),
        dtype=int
    )

    config = Config(
        dir="runs",
        envs=["mnist"],
        seeds=[1],
        launch_script=sys.path[-1] + "tune/examples/script.sh",
        objective_metric="test/accuracy"
    )

    expirement = Study(
        config=config,
        sampler=GridSearchSampler([lr_space, batch_size_space]),
    )
    expirement.start()