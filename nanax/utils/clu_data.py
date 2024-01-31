import rich

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import deterministic_data

def load_dataset(args, seed, print_info=True):

    def make_split(split):

        # TODO: make preprocessing strings
        def preprocess_fn(data, num_epochs=None):
            image = tf.cast(data['image'], tf.float32)
            # image = (image - 127.5) / 127.5
            image = image / 255.
            label = tf.one_hot(data['label'], args.n_classes)
            return {'image': image, 'label': label}

        num_examples = dataset_builder.info.splits[split].num_examples
        split_ds = tfds.split_for_jax_process(split)
        return deterministic_data.create_dataset(
            dataset_builder,
            split=split_ds,
            batch_dims=[jax.local_device_count(), args.batch_size // jax.device_count()],
            rng=seed,
            preprocess_fn=preprocess_fn,
            cache=args.cache_dataset,
            num_epochs=args.num_epochs if split == "train" else None,
            shuffle=True if split == "train" else False,
            shuffle_buffer_size=num_examples,
            prefetch_size=tf.data.experimental.AUTOTUNE,
            drop_remainder=True,
        )

    dataset_builder = tfds.builder(args.dataset_name)
    dataset_builder.download_and_prepare()
    if print_info:
        rich.pretty.pprint(dataset_builder.info)

    return [make_split(split) for split in ["train", "test"]] # TODO: make this dynamic not all datasets have "test"

# @flax.struct.dataclass
# class CluDatasetConfig:
#     split: str = None,
#     batch_dims: Sequence[int] = (),
#     rng: jax.Array = None,
#     filter_fn: ((Features) -> bool) | None = None,
#     preprocess_fn: ((Features) -> Features) | None = None,
#     decoders: Dict[str, Decoder] | None = None,
#     cache: bool = False,
#     num_epochs: int | None = None,
#     shuffle: bool = True,
#     shuffle_buffer_size: int = 10000,
#     prefetch_size: int = 4,
#     pad_up_to_batches: int | str | None = None,
#     cardinality: int | None = None,
#     drop_remainder: bool = True