import os
from math import ceil

import numpy as np
import obonet
import pandas as pd
import tensorflow as tf
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from dlfb.proteins.utils import get_device

import inspect
import re
import textwrap
from itertools import islice
from typing import Callable

from IPython.display import Code
from IPython.display import display as ipython_display

def build_dataset(
    embeddings_dir: Path,  
    model_checkpoint: str
) -> dict:
    dataset_splits = {}

    for split in ["train", "valid", "test"]:
        # Pass the directory explicitly to the loader
        df_split = load_sequence_embeddings(
            store_file_prefix=f"mean_embeddings_{split}", 
            model_checkpoint=model_checkpoint,
            directory=embeddings_dir  
        )

        dataset_splits[split] = convert_to_tfds(
            df=df_split,
            is_training=(split == "train"),
        )
    return dataset_splits

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

def load_sequence_embeddings(
  store_file_prefix: str, model_checkpoint: str
) -> pd.DataFrame:
  """Load stored embedding DataFrame from disk."""
  if os.path.exists(store_file_prefix):
    return pd.read_feather(store_file_prefix)

  model_name = model_checkpoint.replace("/", "_")
  store_file = f"{store_file_prefix}_{model_name}.feather"
  return pd.read_feather(store_file)

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
  
  store_path = os.path.join(directory, f"{store_prefix}_{model_name}.feather")
  
  if not os.path.exists(store_file) or force:
    device = get_device()

    # Iterate through protein dataframe in batches, extracting embeddings.
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
    df.to_feather(store_file)

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

def get_go_term_descriptions(store_path: str) -> pd.DataFrame:
  """Return GO term to description mapping, downloading if needed."""
  if not os.path.exists(store_path):
    url = "https://current.geneontology.org/ontology/go-basic.obo"
    graph = obonet.read_obo(url)

    # Extract GO term IDs and names from the graph nodes.
    id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
    go_term_descriptions = pd.DataFrame(
      zip(id_to_name.keys(), id_to_name.values()),
      columns=["term", "description"],
    )
    go_term_descriptions.to_csv(store_path, index=False)

  else:
    go_term_descriptions = pd.read_csv(store_path)
  return go_term_descriptions

def train_step(state, batch):
  """Run a single training step and update model parametres"""

  def calculate_loss(params):
    """Compute sigmoid cross-entropy loss from logits""" #(unnormalised numerical scores/sigmoid curve loss)
    logits = state.apply_fn({"params":params}, x=batch["embedding"])
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
    return loss

  grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

def eval_step(state, batch) -> dict[str,float]:
  """Run evaluation step. Return mean metrics over the targets"""
  logits=state.apply_fn({"params":state.params}, x=batch["embedding"])
  loss=optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
  target_metrics=calculate_per_target_metrics(logits, batch["target"])
  metrics={"loss":loss.item(), **pd.DataFrame(target_metrics).mean(axis=0).to_dict(),}
  return metrics

def calculate_per_target_metrics(logits, targets):
  """Compute metrics for each target in a multi-label batch"""
  probs=jax.nn.sigmoid(logits)
  target_metrics=[]
  for target, prob in zip(targets, probs):
    target_metrics.append(compute_metrics(target, prob))
  return target_metrics

def train(
  state: TrainState,
  dataset_splits: dict[str, tf.data.Dataset],
  batch_size: int,
  num_steps: int = 600,
  eval_every: int = 30,
):
  """Train model using batched TF datasets and track performance metrics."""
  # Create containers to handle calculated during training and evaluation.
  train_metrics, valid_metrics = [], []

  # Create batched dataset to pluck batches from for each step.
  train_batches = (
    dataset_splits["train"]
    .batch(batch_size, drop_remainder=True)
    .as_numpy_iterator()
  )

  steps = tqdm(range(num_steps))  # Steps with progress bar.
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    # Get batch of training data, convert into a JAX array, and train.
    state, loss = train_step(state, next(train_batches))
    train_metrics.append({"step": step, "loss": loss.item()})

    if step % eval_every == 0:
      # For all the evaluation batches, calculate metrics.
      eval_metrics = []
      for eval_batch in (
        dataset_splits["valid"].batch(batch_size=batch_size).as_numpy_iterator()
      ):
        eval_metrics.append(eval_step(state, eval_batch))
      valid_metrics.append(
        {"step": step, **pd.DataFrame(eval_metrics).mean(axis=0).to_dict()}
      )

  return state, {"train": train_metrics, "valid": valid_metrics}

def make_coin_flip_predictions(
    valid_true_df: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:
  predictions = np.random.choice([0.0, 1.0], size=valid_true_df.shape)
  return pd.DataFrame(predictions, columns=targets, index=valid_true_df.index)

def make_proportional_predictions(
   valid_true_df: pd.DataFrame, train_df: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:

  percent_1_train = dict(train_df[targets].mean())
  proportional_preds = []
  for target_column in targets:
    prob_1 = percent_1_train[target_column]
    prob_0 = 1 - prob_1
    proportional_preds.append(
        np.random.choice([0.0, 1.0], size=len(valid_true_df), p=[prob_0, prob_1])
    )
  return pd.DataFrame(
      np.stack(proportional_preds).T, columns=targets, index=valid_true_df.index
  )

def display(
  sources: list[Callable | str],
  sep: str = "\n\n",
  hide: list[Callable | str] = [],
) -> str:
  blocks = {
    k: [
      source if isinstance(source, str) else inspect.getsource(source)
      for source in v
    ]
    for k, v in {"sources": sources, "hide": hide}.items()
  }
  source_block = sep.join(block for block in blocks["sources"])
  for block in blocks["hide"]:
    source_block = source_block.replace(block, "")
  clean_code = drop_dev_comments(source_block)
  ipython_display(Code(clean_code, language="python"))


def drop_dev_comments(source_block: str):
  filtered_lines = []
  for line in source_block.splitlines():
    if re.match(rf"\s*#\s*{DEV_COMMENT_KEYWORDS}\b", line, re.IGNORECASE):
      continue
    line = re.sub(
      rf"(#\s*){DEV_COMMENT_KEYWORDS}\b.*", "", line, flags=re.IGNORECASE
    )
    filtered_lines.append(line.rstrip())
  return "\n".join(filtered_lines)


def print_short_dict(d, max_items=10, width=80):
  shown = list(islice(d.items(), max_items))
  remaining = len(d) - len(shown)
  preview = {k: v for k, v in shown}
  s = str(preview)
  wrapped_lines = textwrap.wrap(s, width=width)
  for line in wrapped_lines:
    print(line)
  if remaining > 0:
    print(f"…(+{remaining} more entries)")