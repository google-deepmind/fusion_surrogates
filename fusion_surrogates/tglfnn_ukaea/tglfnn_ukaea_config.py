# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config classes for UKAEA's TGLFNN model."""

import dataclasses
import enum
import json
import pathlib
import typing
from typing import Final, Literal, Mapping

import jax
import jax.numpy as jnp
import optax
import torch
import yaml

# Internal import.


Path = pathlib.Path
MEAN_OUTPUT_IDX: Final[int] = 0
VAR_OUTPUT_IDX: Final[int] = 1

OutputLabel = Literal["efe_gb", "efi_gb", "pfi_gb"]
OUTPUT_LABELS = typing.get_args(OutputLabel)


class Machine(enum.Enum):
  STEP = "step"
  MULTIMACHINE = "multimachine"


INPUT_LABELS_DICT: Final[Mapping[Machine, tuple[str, ...]]] = {
    Machine.STEP: (
        "RLNS_1",
        "RLTS_1",
        "RLTS_2",
        "TAUS_2",
        "RMIN_LOC",
        "DRMAJDX_LOC",
        "Q_LOC",
        # Note:
        # SHAT is defined based on Q_PRIME_LOC, rather than from s-α geometry
        "SHAT",
        "XNUE",
        "KAPPA_LOC",
        "S_KAPPA_LOC",
        "DELTA_LOC",
        "S_DELTA_LOC",
        "BETAE",
        "ZEFF",
    ),
    Machine.MULTIMACHINE: (
        "RLNS_1",
        "RLTS_1",
        "RLTS_2",
        "TAUS_2",
        "RMIN_LOC",
        "DRMAJDX_LOC",
        "Q_LOC",
        # Note:
        # SHAT is defined based on Q_PRIME_LOC, rather than from s-α geometry
        "SHAT",
        "XNUE",
        "KAPPA_LOC",
        "DELTA_LOC",
        "ZEFF",
        "VEXB_SHEAR",
    ),
}


@dataclasses.dataclass
class TGLFNNukaeaModelConfig:
  """Config for UKAEA's TGLFNN model."""

  n_ensemble: int
  num_hiddens: int
  dropout: float
  hidden_size: int
  machine: Machine

  @property
  def input_labels(self) -> tuple[str, ...]:
    return INPUT_LABELS_DICT[self.machine]

  @property
  def output_labels(self) -> tuple[str, ...]:
    return OUTPUT_LABELS

  @classmethod
  def load(cls, machine: Machine, config_path: str):
    with open(config_path, "r") as f:
      config = yaml.safe_load(f)

    return cls(
        n_ensemble=config["num_estimators"],
        num_hiddens=config["model_size"],
        dropout=config["dropout"],
        hidden_size=config["hidden_size"],
        machine=machine,
    )


@dataclasses.dataclass
class TGLFNNukaeaModelStats:
  """Stats for UKAEA's TGLFNN model."""

  input_mean: jax.Array
  input_std: jax.Array
  output_mean: jax.Array
  output_std: jax.Array

  @classmethod
  def load(cls, machine: Machine, stats_path: str):
    with open(stats_path, "r") as f:
      stats = json.load(f)

    input_labels = INPUT_LABELS_DICT[machine]
    return cls(
        input_mean=jnp.array([stats[label]["mean"] for label in input_labels]),
        input_std=jnp.array([stats[label]["std"] for label in input_labels]),
        output_mean=jnp.array(
            [stats[label]["mean"] for label in OUTPUT_LABELS]
        ),
        output_std=jnp.array([stats[label]["std"] for label in OUTPUT_LABELS]),
    )


def params_from_pytorch_state_dict(
    pytorch_state_dict: dict[str, torch.Tensor], config: TGLFNNukaeaModelConfig
) -> optax.Params:
  """Converts a PyTorch state dict to an optax Params dict."""
  params = {}
  for i in range(config.n_ensemble):
    model_dict = {}
    for j in range(config.num_hiddens):
      layer_dict = {
          "kernel": (
              jnp.array(
                  pytorch_state_dict[f"models.{i}.model.{j * 3}.weight"]
              ).T
          ),
          "bias": (
              jnp.array(pytorch_state_dict[f"models.{i}.model.{j * 3}.bias"]).T
          ),
      }
      model_dict[f"Dense_{j}"] = layer_dict
    params[f"GaussianMLP_{i}"] = model_dict
  return {"params": params}


def params_from_pt_file(
    checkpoint_path: str | Path,
    config: TGLFNNukaeaModelConfig,
    map_location: str = "cpu",
) -> optax.Params:
  with open(checkpoint_path, "rb") as f:
    return params_from_pytorch_state_dict(
        torch.load(f, map_location=map_location), config
    )
