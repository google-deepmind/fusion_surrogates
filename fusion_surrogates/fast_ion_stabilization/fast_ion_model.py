# Copyright 2026 DeepMind Technologies Limited.
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

"""Inference code for fast ion stabilization models.

These models capture the electrostatic stabilization of ITG turbulence by
ICRH-accelerated fast ions. The effect is quantified as a modification to the
ITG critical gradient threshold: the model predicts the ratio of the ITG
critical gradient with fast ions to the critical gradient without fast ions,
as a function of a subset of fast ion and local plasma parameters.

The training set consists of ITG threshold modifications computed from
gyrokinetic simulations across a range of plasma conditions. The intended
usage is to correct the R/LTi inputs supplied to turbulence transport
surrogates (e.g. QLKNN, TGLFNN), thereby accounting for fast-ion
stabilization without retraining the underlying transport model.

See e.g. A. Di Siena, et al. Physical Review Letters 127.2 (2021): 025002,
for details of the underlying physics.
"""

import dataclasses
from typing import Any, Final

from absl import logging
from flax import serialization
import flax.linen as nn
from fusion_surrogates.common import networks
from fusion_surrogates.common import transforms
from fusion_surrogates.fast_ion_stabilization.models import registry
import jax
import jax.numpy as jnp

# Internal import.
# Internal import.


# TODO(citrin): Consider moving input_features into the model config and
# serializing it in .fistab, to support models with different feature sets.
INPUT_FEATURES: Final[tuple[str, ...]] = (
    'smag',
    'q',
    'n_fi_over_ne',
    't_fi_over_te',
    'rlti_fi',
)
NUM_INPUTS: Final[int] = len(INPUT_FEATURES)
NUM_OUTPUTS: Final[int] = 1
_N_FI_OVER_NE_IDX: Final[int] = INPUT_FEATURES.index('n_fi_over_ne')


class StabilizationMLP(nn.Module):
  """MLP predicting fast ion ITG stabilization factor.

  Output = 1.0 + raw_n_fi_over_ne * MLP(x_normalized).
  Guarantees output = 1.0 when n_fi/ne = 0 (no fast ions).

  Uses common.networks.MLP as the inner torso, with the n_fi gating
  constraint applied on top.
  """

  num_hiddens: int
  hidden_size: int
  activation: str = 'tanh'
  n_fi_over_ne_mean: float = 0.0
  n_fi_over_ne_std: float = 1.0

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    n_fi_over_ne_norm = x[..., _N_FI_OVER_NE_IDX : _N_FI_OVER_NE_IDX + 1]
    n_fi_over_ne_raw = transforms.unnormalize(
        n_fi_over_ne_norm,
        mean=jnp.array(self.n_fi_over_ne_mean),
        stddev=jnp.array(self.n_fi_over_ne_std),
    )
    correction = networks.MLP(
        num_hiddens=self.num_hiddens,
        hidden_size=self.hidden_size,
        num_targets=NUM_OUTPUTS,
        activation=self.activation,
    )(x)
    return 1.0 + n_fi_over_ne_raw * correction


@dataclasses.dataclass
class InputStats:
  """Input feature normalization statistics."""

  mean: jax.Array
  std: jax.Array


@dataclasses.dataclass
class FastIonStabilizationModelConfig:
  """Config for FastIonStabilizationModel."""

  num_hiddens: int
  hidden_size: int
  activation: str
  input_stats: InputStats

  @classmethod
  def deserialize(
      cls, serialized_config: bytes
  ) -> 'FastIonStabilizationModelConfig':
    import_dict = serialization.msgpack_restore(serialized_config)
    if import_dict['input_stats'] is not None:
      import_dict['input_stats'] = InputStats(**import_dict['input_stats'])
    return cls(**import_dict)

  def serialize(self) -> bytes:
    export_dict = dataclasses.asdict(self)
    return serialization.msgpack_serialize(export_dict)


class FastIonStabilizationModel:
  """A JAX fast ion stabilization model.

  Predicts the relative ITG threshold modification factor due to fast ions.
  """

  def __init__(
      self,
      config: FastIonStabilizationModelConfig,
      params: Any | None = None,
      path: str = '',
      name: str = '',
      version: str = '',
      species: str = '',
  ):
    self._config = config
    self._network = StabilizationMLP(
        num_hiddens=config.num_hiddens,
        hidden_size=config.hidden_size,
        activation=config.activation,
        n_fi_over_ne_mean=float(config.input_stats.mean[_N_FI_OVER_NE_IDX]),
        n_fi_over_ne_std=float(config.input_stats.std[_N_FI_OVER_NE_IDX]),
    )
    self._params = params
    self.path = path
    self.name = name
    self.version = version
    self.species = species

  @property
  def config(self) -> FastIonStabilizationModelConfig:
    return self._config

  @property
  def network(self) -> nn.Module:
    return self._network

  @property
  def params(self) -> Any:
    if self._params is None:
      raise ValueError('Params have not been initialized.')
    return self._params

  @params.setter
  def params(self, params: Any) -> None:
    self._params = params

  def predict(self, inputs: jax.Array) -> jax.Array:
    """Predicts the stabilization factor from raw (unnormalized) inputs.

    Input normalization is handled internally using stored input statistics.
    No output denormalization is needed: the network directly outputs a
    physically meaningful ratio (the ITG threshold modification factor, ≈ 1.0),
    constrained by the architecture (output = 1.0 + n_fi/ne * MLP(x)).

    Args:
      inputs: Array of shape (..., NUM_INPUTS) with raw input features in the
        order defined by INPUT_FEATURES.

    Returns:
      Array of shape (..., 1) with the relative ITG threshold factor.
    """
    if self._params is None:
      raise ValueError('Params have not been initialized.')
    normalized = transforms.normalize(
        inputs,
        mean=jnp.array(self._config.input_stats.mean),
        stddev=jnp.array(self._config.input_stats.std),
    )
    return self._network.apply(self._params, normalized)

  def export_model(self, output_path: str) -> None:
    """Exports the model to a .fistab file."""
    export_dict = {
        'version': self.version,
        'config': self._config.serialize(),
        'params': self._params,
        'species': self.species,
    }
    with open(output_path, 'wb') as f:
      f.write(serialization.msgpack_serialize(export_dict))

  @classmethod
  def load_model_from_name(cls, model_name: str) -> 'FastIonStabilizationModel':
    """Loads a FastIonStabilizationModel by name from the registry."""
    model_path = registry.MODELS.get(model_name)
    if model_path is None:
      raise ValueError(f'Model {model_name} not found in registry.')
    return cls.load_model_from_path(model_path, model_name)

  @classmethod
  def load_model_from_path(
      cls,
      input_path: str,
      model_name: str | None = None,
  ) -> 'FastIonStabilizationModel':
    """Loads a FastIonStabilizationModel from a file path."""
    logging.info('Loading FastIonStabilizationModel from %s', input_path)
    with open(input_path, 'rb') as f:
      import_dict = serialization.msgpack_restore(f.read())
    if model_name is None:
      model_name = ''
    return cls(
        config=FastIonStabilizationModelConfig.deserialize(
            import_dict['config']
        ),
        params=import_dict['params'],
        path=input_path,
        name=model_name,
        version=import_dict['version'],
        species=import_dict.get('species', ''),
    )

  @classmethod
  def load_default_model_for_species(
      cls, species: str
  ) -> 'FastIonStabilizationModel':
    """Loads the default FastIonStabilizationModel for the given species."""
    model_name = registry.DEFAULT_MODELS.get(species)
    if model_name is None:
      raise ValueError(
          f'No default model for species {species!r}.'
          f' Available: {list(registry.DEFAULT_MODELS.keys())}'
      )
    return cls.load_model_from_name(model_name)
