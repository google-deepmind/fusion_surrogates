"""Base code for UKAEA's TGLFNN model.

UKAEA-TGLFNN is a neural network surrogate model for the gyrokinetics code TGLF, developed by Lorenzo Zanisi at UKAEA.
The model is trained on a dataset generated from JETTO TGLF runs in the STEP design space.
Hence, it is best suited to modelling transport in spherical tokamaks.
"""

import dataclasses
import json
import typing
from typing import Literal


import jax
import jax.numpy as jnp
import optax
import yaml

from fusion_surrogates import networks
from fusion_surrogates import transforms

OutputLabel = Literal["efe_gb", "efi_gb", "pfi_gb"]
OUTPUT_LABELS = typing.get_args(OutputLabel)

InputLabel = Literal[
    "RLNS_1",
    "RLTS_1",
    "RLTS_2",
    "TAUS_2",
    "RMIN_LOC",
    "DRMAJDX_LOC",
    "Q_LOC",
    "SHAT",
    "XNUE",
    "KAPPA_LOC",
    "S_KAPPA_LOC",
    "DELTA_LOC",
    "S_DELTA_LOC",
    "BETAE",
    "ZEFF",
]
INPUT_LABELS = typing.get_args(InputLabel)


@dataclasses.dataclass
class TGLFNNModelConfig:
    n_ensemble: int
    num_hiddens: int
    dropout: float
    normalize: bool = True
    unnormalize: bool = True
    hidden_size: int = 512

    @classmethod
    def load(cls, config_path: str) -> "TGLFNNModelConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(
            n_ensemble=config["num_estimators"],
            num_hiddens=config["model_size"],
            dropout=config["dropout"],
        )


@dataclasses.dataclass
class TGLFNNModelStats:
    input_mean: jax.Array
    input_std: jax.Array
    output_mean: jax.Array
    output_std: jax.Array

    @classmethod
    def load(cls, stats_path: str) -> "TGLFNNModelStats":
        with open(stats_path, "r") as f:
            stats = json.load(f)

        return cls(
            input_mean=jnp.array([stats[label]["mean"] for label in INPUT_LABELS]),
            input_std=jnp.array([stats[label]["std"] for label in INPUT_LABELS]),
            output_mean=jnp.array([stats[label]["mean"] for label in OUTPUT_LABELS]),
            output_std=jnp.array([stats[label]["std"] for label in OUTPUT_LABELS]),
        )
