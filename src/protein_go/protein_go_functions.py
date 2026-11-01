# Placeholder

import pandas as pd
import tensorflow as tf
import numpy as np


def convert_to_tfds(
    df: pd.DataFrame,
    embeddings_prefix: str = "ME:",
    target_prefix: str = "GO:",
    is_training: bool = False,
    shuffle_buffer: int = 50,
) -> tf.data.Dataset:
    """Convert embedding DataFrame into a TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "embedding": df.filter(regex=f"^{embeddings_prefix}").to_numpy(),
            "target": df.filter(regex=f"^{target_prefix}").to_numpy(),
        }
    )
    if is_training:
        dataset = dataset.shuffle(shuffle_buffer).repeat()
    return dataset


##################################

from functools import wraps
from pathlib import Path

import jax
import orbax.checkpoint as ocp

import logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(module)s - %(funcName)s: "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

logging.getLogger("absl").addFilter(lambda _: False)
logging.getLogger("matplotlib").setLevel("WARNING")


def restorable(train_fn):
    # NOTE: wraps required for proper display with 'inspect'
    @wraps(train_fn)
    def wrapper(state, store_path: str | None = None, **kwargs):
        if store_path:
            mngr = ocp.CheckpointManager(Path(store_path).resolve())
            try:
                state, metrics = restore_checkpoint(mngr, state)
            except FileNotFoundError:
                log.debug("Training new model checkpointing state and metrics...")
                state, metrics = train_fn(state, **kwargs)
                save_final_checkpoint(mngr, state, metrics)
        else:
            log.debug("Training new model without checkpointing...")
            state, metrics = train_fn(state, **kwargs)
        return state, metrics

    return wrapper


def restore_checkpoint(mngr: ocp.CheckpointManager, state):
    # NOTE: benefit of approach is that we can reconstruct a model
    #       built on a GPU back in a CPU environment.
    #       see: https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpointing_pytrees.html#change-sharding
    restored = mngr.restore(
        0,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(get_abstract_state(state)),
            extra_metadata=ocp.args.JsonRestore(),
        ),
    )
    log.debug("Train state and metrics restored from checkpoint")
    state, metrics = restored.state, restored.extra_metadata
    return state, metrics


def save_final_checkpoint(mngr, state, metrics):
    mngr.save(
        0,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            extra_metadata=ocp.args.JsonSave(metrics),
        ),
    )
    mngr.wait_until_finished()


def get_abstract_state(state):
    """Returns a shape/dtype-only version of the model's state"""
    return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
