import os
import re
import json
import time
import threading
import subprocess
from dataclasses import dataclass
from typing import List, Callable, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

MAX_RETRIES = 5
RETRY_DELAY = 10
MONITOR_DELAY = 1

@dataclass
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


# create RL Trial and DL trial
class Trial:

    def __init__(self, args, config, callback) -> None:
        self.args = args
        self.config = config
        self.callback = callback
        self.jobs = {}

    def start(self):
        for env in self.config.envs:
            if env not in self.jobs:
                self.jobs[env] = {}
            for seed in self.config.seeds:
                job = self._start_job(env, seed)
                self.jobs[env][seed] = job
                self.callback(job)
                if self.config.verbose:
                    print(f"Submitted job {job.job_id} for args {self.args_str_id()} on {env} with seed {seed}")

    def is_running(self):
        return all(self.jobs[env][seed].is_running() for env in self.config.envs for seed in self.config.seeds)

    def is_finished(self):
        return all(self.jobs[env][seed].is_finished() for env in self.config.envs for seed in self.config.seeds)

    def was_pruned(self):
        return False

    def args_str_id(self):
        return "-".join([f"{key}={value}" for key, value in self.args.items()])

    def compute_summary_metric(self):
        event_files = self._load_event_files()

        metrics = {}
        index = None
        for env in self.config.envs:
            seeds = []
            for seed in self.config.seeds:
                metric = event_files[env][seed].get_objective_metric()
                if index is None:
                    index = metric['step']
                seeds.append(pd.Series(metric['value'], index=metric['step']))

            df = pd.concat(seeds, axis=1)
            metrics[env] = {
                'mean': df.mean(axis=1).tolist(),
                'std': df.std(axis=1).tolist()
            }

        envs = pd.DataFrame({env: pd.Series(metrics[env]['mean']) for env in self.config.envs})
        # here you will calculate the normalized score for each environment then take the mean over all environments

        mean = envs.mean(axis=1).tolist()
        std = envs.std(axis=1).tolist()

        return index, mean

    def _start_job(self, env, seed):
        args = self._create_job_specific_args(env, seed)
        job = Job(args, self.config.launch_script)
        job.start()
        return job

    def _create_job_specific_args(self, env, seed):
        args = self.args.copy()
        args['seed'] = seed
        if self.config.include_env: # TODO: this is a hack
            args['env_id'] = env
        args['tensorboard_dir'] = f"{self.config.dir}/{self.args_str_id()}/env={env}/seed={seed}/"
        return args

    def _load_event_files(self):
        event_files = {}
        for env in self.config.envs:
            event_files[env] = {}
            for seed in self.config.seeds:
                job = self.jobs[env][seed]
                dir = job.args['tensorboard_dir']
                file = os.path.join(dir, os.listdir(dir)[0])
                event_files[env][seed] = EventFile(file, self.config.objective_metric)
        return event_files


# as you write trials to results
# cache those results so that if you restart the monitor it can load the results from disk and keep running other trials

class Monitor:

    def __init__(self, dir, trials, verbose):
        self.dir: str = dir + "/results"
        self.cache_dir = self.dir + "/cache"
        self.trials: List = trials
        self.verbose = verbose

    def start(self):
        self.thread = threading.Thread(
            target=self._monitor_trials
        )
        self.thread.start()

    # when metrics starts to get big you'll need to write it to disk then iterate over it when you add it to the plot
    def _monitor_trials(self):
        os.makedirs(self.dir, exist_ok=True)
        self.metrics, self.index = self._load_cache()
        self.has_started = False
        shutdown_counter = 0
        while True:

            time.sleep(MONITOR_DELAY)
            if len(self.trials) == 0 and not self.has_started:
                continue
            
            elif not self.has_started:
                self.has_started = True

            try:
                self._calculate_trials_summary_metric() 
                self.update_plot()
            except Exception as e:
                if "available metrics are []" in str(e):
                    # trials have started but tf events have not been written to yet
                    continue
                elif "No such file or director" in str(e):
                    continue
                print(f"MONITOR: Failed to update plot: {e}")

    def update_plot(self):
        plt.figure()
        for config_str, metric in self.metrics.items():
            plt.plot(self.index, metric, label=config_str)

        plt.title("Study")
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        # plt.yscale('logit')
        plt.legend(title="Batch Size", loc=4)
        plt.grid(True)
        plt.savefig(f"{self.dir}/plot.png")
        plt.close()

    def _load_cache(self):
        # if a cache does not exist, create one
        if not os.path.isfile(f"{self.cache_dir}/metrics.json"):
            return {}, None
        else:
            with open(f"{self.cache_dir}/metrics.json", "r") as f:
                cache = json.load(f)
            metrics = {}
            index_match = None
            for config_str, data in cache.items():
                metrics[config_str] = data['metric']
                if index_match is None:
                    index_match = data['index']
                index = data['index']
                assert index_match == index, "Index mismatch in cache, manual inspection required."

            return metrics, index

    def _calculate_trials_summary_metric(self):
        remove_trials = []
        for trial in self.trials:
            if trial.was_pruned():
                continue
            index, metric = trial.compute_summary_metric()
            self.metrics[trial.args_str_id()] = metric

            if trial.is_finished() and len(metric) > 0:
                self._write_to_cache(trial.args_str_id(), index, metric)
                remove_trials.append(trial)

        if self.index is None:
            self.index = index
            # raise Exception("No trials to monitor")

        if len(remove_trials) > 0:
            if self.verbose:
                print(f"MONITOR: Removing {len(remove_trials)} trials from monitoring.")
            for trial in remove_trials:
                self.trials.remove(trial)

    def _write_to_cache(self, config_str, index, metric):
        os.makedirs(self.cache_dir, exist_ok=True)

        # look for metrics.json
        if not os.path.isfile(f"{self.cache_dir}/metrics.json"):
            with open(f"{self.cache_dir}/metrics.json", "w") as f:
                f.write("{}")

        # load metrics.json as json
        with open(f"{self.cache_dir}/metrics.json", "r") as f:
            cache = json.load(f)

        cache.update({config_str: {'index': index, 'metric': metric}})
        # write payload to metrics.json as json
        with open(f"{self.cache_dir}/metrics.json", "w") as f:
            f.write(json.dumps(cache))

    def load_results(self):
        metrics, _ = self._load_cache()
        return metrics

    def get_best_args(self, best_obective_metric=None, iteration=None):
        metrics, index = self._load_cache()
        best_objective_metric = 0 if best_obective_metric == None else best_obective_metric
        best_args = None
        for config_str, metric in metrics.items():
            if metric[-1] > best_objective_metric:
                best_objective_metric = metric[-1]
                best_args = config_str
        # iteration = str(iteration) if not None else ''
        # print(f"MONITOR: {iteration} | best args: {best_args}, with objective metric of {best_objective_metric}")
        return best_args, best_obective_metric

    def stop(self):
        self.thread.join()


class SlurmManager:
    """
    Maintains a list of active Jobs and updates their status.
    """
    
    def __init__(self) -> None:
        self.active_jobs = []

    def start(self):
        self.thread = threading.Thread(
            target=self.manage_active_jobs
        )
        self.thread.start()

    def stop(self):
        self.thread.join()

    def manage_active_jobs(self):
        while True:
            if len(self.active_jobs) == 0:
                # print("MANAGER: No active jobs.")
                time.sleep(10)
            
            for job in self.active_jobs:
                job.update_status()
                if job.is_finished():
                    self.active_jobs.remove(job)
                    continue
                if job.is_failed():
                    print(f"Job {job.job_id} failed. Status: {job.status}")
                    print("Going to attempt to resubmit job.")
                    job.start()
                    job.update_status()
                    print(f"RESTART: Job {job.job_id}. Status: {job.status}")
                # if not job.is_running() or not job.is_pending():
                #     print(f"MANAGER: Job {job.job_id} is not running, status: {job.status}")

    def add_job(self, job):
        self.active_jobs.append(job)


class Job:
    """
    Submits a job to a Slurm cluster and tracks its status.
    """

    def __init__(self, args, launch_script) -> None:
        self.args = args
        self.launch_script = launch_script
        self.job_id = None
        self.status = None

    def start(self):
        args_str = self._make_args_str()
        output = self._execute_with_retries(self._sumbit_job, args_str)
        self.job_id = self._parse_job_id(output)
        self.update_status()
        print(f"Sumbitted {self.job_id} with status: {self.status}")

    def update_status(self):
        self._execute_with_retries(self._update_status)

    def is_pending(self):
        return self.status == "PENDING"

    def is_running(self):
        return self.status == "RUNNING"
    
    def is_failed(self):
        return self.status not in ["PENDING", "RUNNING", "COMPLETED", "COMPLETING"]

    def is_finished(self):
        return self.status == "COMPLETED"

    def _execute_with_retries(self, func: Callable, *args):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args)
            except Exception as e:
                print(f"Attempt {attempt + 1}: Failed to execute '{func.__name__}', {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise Exception(f"Max retries reached. Failed to execute '{func.__name__}'.")

    def _update_status(self):
        result = subprocess.run(["squeue", "-h", "-j", self.job_id, "-o", "%T"], capture_output=True)
        status = result.stdout.decode().strip()
        valid_statuses = [
            "COMPLETED", "COMPLETING", "FAILED", "PENDING",
            "PREEMPTED", "RUNNING", "SUSPENDED", "STOPPED"]
        if status not in valid_statuses:
            if status == "":
                # raise ValueError(f"Job {self.job_id} not found.")
                status = "COMPLETED"
            else:
                raise ValueError(f"Unknown job status: {status}")
        self.status = status

    def _sumbit_job(self, args):
        return subprocess.run(["sbatch", self.launch_script, args], capture_output=True)

    def _parse_job_id(self, output):
        stdout = output.stdout.decode()
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            if stdout == "":
                raise Exception("Failed to submit job. stdout is empty. Are you sure you passed the right path to the launch script?")
            else:
                raise Exception(f"Failed to submit job or extract job ID. stdout: {stdout}")
        return match.group(1)

    def _make_args_str(self):
        return " ".join([f"--{k} {v}" for k, v in self.args.items()])


class EventFile:
    """
    Reads Tensorboard event files and convert them to Pandas DataFrames.

    Args:
        path: Path to the event file.
        objective_metric: The metric being optimized in the hyperparameter search.
    """

    def __init__(self, path: str, objective_metric: str) -> None:
        self.path = path
        self.objective_metric = objective_metric

    def get_objective_metric(self):
        data = self._read_event_file()
        try:
            return data[self.objective_metric]
        except KeyError:
            raise KeyError(f"Objective metric {self.objective_metric} not found in event file {self.path}, available metrics are {list(data.keys())}")

    def _read_event_file(self):
        if not os.path.isfile(self.path):
            return {}
        
        data = {}
        for record in tf.data.TFRecordDataset(self.path):
            event = tf.compat.v1.Event.FromString(record.numpy())
            for v in event.summary.value:
                if "hyperparameters" in v.tag and v.metadata.plugin_data.plugin_name == 'text':
                    hyperparameters_str = v.tensor.string_val.pop().decode() # TODO: delete or do something with it
                elif v.HasField('simple_value'): # scalar data
                    if v.tag not in data:
                        data[v.tag] = {'step': [], 'value': []}
                    data[v.tag]['step'].append(event.step)
                    data[v.tag]['value'].append(v.simple_value)
        return data

    def convert_to_dataframes(self):
        data = self._read_event_file()
        for tag in data:
            df = pd.DataFrame(data[tag])
            df.columns = [f"{tag}_{col}" for col in df.columns]
            data[tag]['df'] = df

        return data