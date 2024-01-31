import jax
import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass, field
from datasets import Dataset, load_dataset

@dataclass
class Dataset:
    dataset: str
    train_file: str = None
    validation_file: str = None
    streaming: bool = True
    preprocessing_num_workers: int = None
    overwrite_cache: bool = False
    seed_dataset: int = 42 * 100003 # Prime
    shard_by_host: bool = False
    rng_dataset: jnp.ndarray = field(init=False)
    multi_hosts: bool = field(init=False)
    use_auth_token: bool = False
    remove_columns: list = field(default_factory=lambda: [])

    def __post_init__(self):
        self.np_rng = np.random.default_rng(self.seed_dataset)
        self.multi_hosts = jax.process_count() > 1

        dataset = load_dataset(
            self.dataset,
            streaming=self.streaming,
            use_auth_token=self.use_auth_token,
        )
        self.train_dataset = dataset['train']
        self.test_dataset = dataset['test']

    def preprocess(self):
        if self.streaming:
            self.train_dataset = self.train_dataset.shuffle(
                buffer_size=5000,
                seed=self.seed_dataset
            )
            self.train_dataset.map(
                function=preprocess_fn,
                batched=True,
                remove_columns=self.remove_columns,
            )
            self.test_dataset.map(
                function=preprocess_fn,
                batched=True,
            )
        else:
            self.rng_dataset = jax.random.PRNGKey(self.seed_dataset)
            self.train_dataset.map(
                function=preprocess_fn,
                batched=True,
                remove_columns=self.remove_columns,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
                desc="Preprocessing datasets",
            )

    def dataloader(self, split, batch_size, epoch=None):

        def _dataloader_non_streaming(dataset: Dataset, rng: jax.Array = None):
            steps_per_epoch = len(dataset) // batch_size
            if rng is not None:
                batch_idx = jax.random.permutation(rng, len(dataset))
            else:
                batch_idx = jnp.arange(len(dataset))

            batch_idx = batch_idx[: steps_per_epoch * batch_size]
            batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

            for idx in batch_idx:
                batch = dataset[idx]
                batch = {k: jnp.array(v) for k, v in batch.items()}
                yield batch

        def _dataloader_streaming(self, dataset: Dataset, epoch: int):
            keys = ['image', 'labels']
            batch = {k: [] for k in keys}
            first_loop = True  # stop after one loop in some cases
            while (self.multi_hosts and split == "train") or first_loop:
                # in multi-host, we run forever (no epoch) as hosts need to stop
                # at the same time and training data may not be split equally
                # For validation data we put the entire batch on each host and then
                # keep only the one specific to each host (could be improved but not necessary)
                if epoch is not None:
                    assert split == "train"
                    # reshuffle training data at each epoch
                    dataset.set_epoch(epoch)
                    epoch += 1
                for item in dataset:
                    for k in keys:
                        batch[k].append(item[k])
                    if len(batch[keys[0]]) == batch_size:
                        batch = {k: jnp.array(v) for k, v in batch.items()}
                        yield batch
                        batch = {k: [] for k in keys}
                first_loop = False

        if split == "train":
            ds = self.train_dataset
        elif split == "eval":
            ds = self.eval_dataset
        else:
            ds = self.other_eval_datasets[split]

        if self.streaming:
            return _dataloader_non_streaming(ds, epoch)
        else:
            if split == "train":
                self.rng_dataset, input_rng = jax.random.split(self.rng_dataset)
            return _dataloader_streaming(ds, input_rng)


    @property
    def length(self):
        len_train_dataset, len_eval_dataset = None, None
        if self.streaming:
            # we don't know the length, let's just assume max_samples if defined
            if self.max_train_samples is not None:
                len_train_dataset = self.max_train_samples
            if self.max_eval_samples is not None:
                len_eval_dataset = self.max_eval_samples
        else:
            len_train_dataset = (
                len(self.train_dataset) if hasattr(self, "train_dataset") else None
            )
            len_eval_dataset = (
                len(self.eval_dataset) if hasattr(self, "eval_dataset") else None
            )
        return len_train_dataset, len_eval_dataset

def preprocess_fn(example):
    image = np.array(example["image"])
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return {"image": image, "label": example["label"]}

if __name__ == "__main__":
    dataset = Dataset(
        dataset="mnist",
        train_file="train",
        validation_file="test",
        streaming=True,
        preprocessing_num_workers=4,
    )
    dataset.preprocess()
    dataloader = dataset.dataloader("train", 32)
    print(dataloader)