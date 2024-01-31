def load_hf_dataset_splits(dataset_name, streaming=False, use_auth_token=False):

    def prefetch(dataset, n_prefetch):
        """Prefetches data to device and converts to numpy array."""
        ds_iter = iter(dataset)
        ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                    ds_iter)
        if n_prefetch:
            ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
        return ds_iter
 
    def load_split(split):
        """Load dataset from huggingface datasets."""
        
        def _preprocess(data):
            image = tf.cast(data['image'], tf.float32)
            # image = (image - 127.5) / 127.5
            image = image / 255.
            image = tf.expand_dims(image, -1)
            label = tf.one_hot(data['label'], args.n_classes)
            return {'image': image, 'label': label}

        # data = data[::jax.num_devices()], TODO: still need how to figure out how to shard across nodes

        # load dataset and convert to tf dataset
        data = load_dataset(dataset_name, split=split, streaming=streaming, use_auth_token=use_auth_token)
        num_examples = data.num_rows
        data = data.to_tf_dataset(
            columns=["image", "label"],
            batch_size=args.batch_size,
            shuffle=True if split == "train" else False,
            prefetch=True,
            drop_remainder=True,
            # collate_fn=None, 
        ).repeat()

        # apply preprocessing
        data = data.map(
            _preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=True
        )

        # shard data across devices
        num_devices = jax.local_device_count()
        def _shard(data, image_shape, num_classes):
            data['image'] = tf.reshape(data['image'], [num_devices, -1, *image_shape])
            data['label'] = tf.reshape(data['label'], [num_devices, -1, num_classes])
            return data
        _shard = partial(
            _shard,
            image_shape=data.element_spec['image'].shape[1:],
            num_classes=args.n_classes
        )
        data = data.map(_shard, tf.data.experimental.AUTOTUNE)

        return prefetch(data, 2), num_examples // args.batch_size # TODO: this is not correct for distributed training

    train_dataset, updates_per_epoch = load_split("train")
    test_dataset, eval_iters = load_split("test")

    return train_dataset, test_dataset, updates_per_epoch, eval_iters