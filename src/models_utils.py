import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import tensorflow as tf
import sklearn.metrics as skm
import pandas as pd
import os
import optax
import torch
import math

from flax.training.train_state import TrainState
from transformers import PreTrainedTokenizer, PreTrainedModel
from pathlib import Path
from src.protein_go.protein_go_functions import convert_to_tfds
from math import ceil


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def store_sequence_embeddings(
    sequence_df: pd.DataFrame,
    store_prefix: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    directory: str,
    batch_size: int = 64,
    force: bool = False,
) -> None:
    """Extract and store mean embeddings for each protein sequence."""

    model_name = model.config._name_or_path.split("/")[-1]
    store_path = os.path.join(directory, f"{store_prefix}_{model_name}.feather")

    if not os.path.exists(store_path) or force:
        device = get_device()

        n_batches = ceil(sequence_df.shape[0] / batch_size)
        batches: list[np.ndarray] = []
        for i in range(n_batches):
            batch_seqs = list(
                sequence_df["Sequence"][i * batch_size : (i + 1) * batch_size]
            )
            batches.extend(get_mean_embeddings(batch_seqs, tokenizer, model, device))

        # Store each of the embedding values in a separate column in the dataframe.
        embeddings = pd.DataFrame(np.vstack(batches))
        embeddings.columns = [f"ME:{int(i)+1}" for i in range(embeddings.shape[1])]
        df = pd.concat([sequence_df.reset_index(drop=True), embeddings], axis=1)
        df.to_feather(store_path)


def get_mean_embeddings(
    sequences: list[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute mean embedding for each sequence using a protein LM."""
    if not device:
        device = get_device()

    # Tokenize input sequences and pad them to equal length.
    model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

    # Move tokenized inputs to the target device (CPU or GPU).
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # Move model to the target device and set it to evaluation mode.
    model = model.to(device)
    model.eval()

    # Forward pass without gradient tracking to obtain embeddings.
    with torch.no_grad():
        outputs = model(**model_inputs)
        mean_embeddings = outputs.last_hidden_state.mean(dim=1)

    return mean_embeddings.detach().cpu().numpy()


def load_sequence_embeddings(
    store_file_prefix: str, model_checkpoint: str, directory: Path  # Add this!
) -> pd.DataFrame:
    """Load stored embedding DataFrame from disk."""
    model_name = model_checkpoint.replace("/", "_")
    filename = f"{store_file_prefix}_{model_name}.feather"

    full_path = directory / filename

    if not full_path.exists():
        raise FileNotFoundError(f"Could not find embedding file at: {full_path}")

    return pd.read_feather(full_path)


def build_dataset(embeddings_dir: Path, model_checkpoint: str) -> dict:
    dataset_splits = {}

    for split in ["train", "valid", "test"]:

        df_split = load_sequence_embeddings(
            store_file_prefix=f"mean_embeddings_{split}",
            model_checkpoint=model_checkpoint,
            directory=embeddings_dir,
        )

        dataset_splits[split] = convert_to_tfds(
            df=df_split,
            is_training=(split == "train"),
        )
    return dataset_splits


class ProteinMLP(nn.Module):
    """Simple MLP for protein function prediction"""

    num_targets: int
    dim: int = 1024

    @nn.compact
    def __call__(self, x, train: bool = True):
        """Apply MLP layers to input features"""
        # Here I added an extra layer and also dropout

        # Layer 1
        x = nn.Dense(self.dim * 2)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        # Layer 2
        x = nn.Dense(self.dim * 2)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        # Layer 2
        x = nn.Dense(self.dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        # Output
        x = nn.Dense(self.num_targets)(x)
        return x

    def create_train_state(self, rng: jax.Array, dummy_input, tx) -> TrainState:
        """Initialise the model and return the training state."""
        variables = self.init(rng, dummy_input, train=False)
        return TrainState.create(apply_fn=self.apply, params=variables["params"], tx=tx)


def train_step(state, batch):
    """Run a single training step and update model parametres"""

    def calculate_loss(params):
        """Compute sigmoid cross-entropy loss from logits"""  # (unnormalised numerical scores/sigmoid curve loss)
        logits = state.apply_fn({"params": params}, x=batch["embedding"])
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
        return loss

    grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_metrics(
    targets: np.ndarray, probs: np.ndarray, thresh=0.5
) -> dict[str, float]:

    targets = np.array(targets)
    probs = np.array(probs)

    if np.sum(targets) == 0:
        return {m: 0.0 for m in ["accuracy", "recall", "precision", "auprc", "auroc"]}

    return {
        "accuracy": float(skm.accuracy_score(targets, probs >= thresh)),
        "recall": float(skm.recall_score(targets, probs >= thresh, zero_division=0.0)),
        "precision": float(
            skm.precision_score(targets, probs >= thresh, zero_division=0.0)
        ),
        "auprc": float(skm.average_precision_score(targets, probs)),
        "auroc": float(skm.roc_auc_score(targets, probs)),
    }
