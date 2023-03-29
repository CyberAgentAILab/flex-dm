import gc
import logging
import os
from datetime import datetime, timezone

import tensorflow as tf

logger = logging.getLogger(__name__)


class HyperTune(tf.keras.callbacks.Callback):
    """Callback for HyperTune on AI Platform."""

    def __init__(self, metric, tag=None, logdir=None, **kwargs):
        super().__init__(**kwargs)
        self._metric = metric
        self._tag = tag or "training/hptuning/metric"
        self._logdir = logdir or "/tmp/hypertune/output.metrics"
        self._writer = tf.summary.create_file_writer(self._logdir)

    def on_epoch_end(self, epoch, logs=None):
        if logs and self._metric in logs:
            with self._writer.as_default():
                tf.summary.scalar(self._tag, logs[self._metric], step=epoch)
            now = datetime.now(timezone.utc).astimezone().isoformat()
            print(f"{now} {self._tag} = {logs['val_loss']}")


class GarbageCollector(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def get_callbacks(args, dataspec, checkpoint_path: str):
    log_dir = os.path.join(args.job_dir, "logs")
    if tf.io.gfile.exists(log_dir):
        logger.warning("Overwriting log dir: %s" % log_dir)
        tf.io.gfile.rmtree(log_dir)

    logger.info(f"checkpoint_path={checkpoint_path}")
    logger.info(f"log_dir={log_dir}")

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=False,
        profile_batch=2 if args.enable_profile else 0,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        monitor="val_total_score",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
    gc = GarbageCollector()
    callbacks_list = [tensorboard, checkpoint, terminate_on_nan, gc]
    return callbacks_list
