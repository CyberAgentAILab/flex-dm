import json
import logging
import os
import random

import numpy as np
import tensorflow as tf
from fsspec.core import url_to_fs
from mfp.data import DataSpec
from mfp.helpers.callbacks import get_callbacks
from mfp.models.mfp import MFP

logger = logging.getLogger(__name__)


def train(args):
    logger.info(f"tensorflow version {tf.__version__}")
    # fix seeds for reproducibility and stable validation
    seed = args.seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # strategy = tf.distribute.MirroredStrategy()
    fs, _ = url_to_fs(args.job_dir)
    if not fs.exists(args.job_dir):
        fs.makedir(args.job_dir)

    json_path = os.path.join(args.job_dir, "args.json")

    with fs.open(json_path, "w") as file_obj:
        json.dump(vars(args), file_obj, indent=2)
    checkpoint_dir = os.path.join(args.job_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best.ckpt")

    dataspec = DataSpec(
        args.dataset_name,
        args.data_dir,
        batch_size=args.batch_size,
    )

    train_dataset = dataspec.make_dataset(
        "train",
        shuffle=True,
        repeat=True,
        cache=True,
    )
    val_dataset = dataspec.make_dataset("val", cache=True)
    test_dataset = dataspec.make_dataset("test", cache=True)

    input_columns = dataspec.make_input_columns()
    model = MFP(
        input_columns,
        num_blocks=args.num_blocks,
        block_type=args.block_type,
        masking_method=args.masking_method,
        seq_type=args.seq_type,
        arch_type=args.arch_type,
        context=args.context,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        l2=args.l2,
        input_dtype=args.input_dtype,
    )

    if args.weights:
        logger.info("Loading %s" % args.weights)
        model.load_weights(args.weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            clipnorm=1.0,
        ),
        run_eagerly=True,
    )

    model.fit(
        train_dataset,
        steps_per_epoch=dataspec.steps_per_epoch("train"),
        epochs=args.num_epochs,
        validation_data=val_dataset,
        validation_steps=dataspec.steps_per_epoch("val"),
        validation_freq=min(args.validation_freq, args.num_epochs),
        callbacks=get_callbacks(args, dataspec, checkpoint_path),
        verbose=args.verbose,
    )

    results = model.evaluate(test_dataset, batch_size=args.batch_size)
    for k, v in zip(model.metrics_names, results):
        print(k, v)

    # Save the last model.
    model_path = os.path.join(args.job_dir, "checkpoints", "final.ckpt")
    logger.info("Saving %s" % model_path)
    model.save_weights(model_path)
