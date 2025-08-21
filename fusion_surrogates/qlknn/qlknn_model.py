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

"""Inference code for QLKNN models.

QLKNN stands for QuaLiKiz Neural Network. These are surrogate models of data
generated from QuaLiKiz, a quasilinear gyrokinetic model used to compute
turbulent transport in plasmas.
"""
import abc
from collections.abc import Mapping
import dataclasses
import json
from typing import Any, Final

from absl import logging
from flax import serialization
from flax import typing as flax_typing
import flax.linen as nn
from fusion_surrogates.common import networks
from fusion_surrogates.common import transforms
from fusion_surrogates.qlknn.models import registry
import immutabledict
import jax
import jax.numpy as jnp
import optax

# Internal import.
# Internal import.


VERSION: Final[str] = '11D'


@dataclasses.dataclass
class QLKNNStatsData:
  """Stats data for normalization in QLKNNModel."""

  input_mean: jax.Array
  input_stddev: jax.Array
  target_mean: jax.Array
  target_stddev: jax.Array
  input_min: jax.Array
  input_max: jax.Array

  @classmethod
  def from_stats_path(
      cls, stats_path: str, num_inputs: int
  ) -> 'QLKNNStatsData':
    logging.info('Loading data stats from %s', stats_path)
    with open(stats_path, 'r') as f:
      data_stats = json.load(f)
    return cls(
        input_mean=jnp.array(data_stats['mean'][:num_inputs]),
        input_stddev=jnp.array(data_stats['stddev'][:num_inputs]),
        target_mean=jnp.array(data_stats['mean'][num_inputs:]),
        target_stddev=jnp.array(data_stats['stddev'][num_inputs:]),
        input_min=jnp.array(data_stats['min'][:num_inputs]),
        input_max=jnp.array(data_stats['max'][:num_inputs]),
    )


@dataclasses.dataclass
class NetworkConfig(abc.ABC):
  """Config for a neural network."""


@dataclasses.dataclass
class MLPConfig(NetworkConfig):
  """Config for an MLP."""

  num_hiddens: int
  hidden_size: int
  activation: str


@dataclasses.dataclass
class CGMConfig(NetworkConfig):
  """Config for a Critical Gradient Model Network."""

  torso_num_hiddens: int
  torso_hidden_size: int
  head_num_hiddens: int
  head_hidden_size: int
  activation: str


NETWORK_CONFIG_MAP: Final[
    Mapping[networks.NetworkType, type[NetworkConfig]]
] = immutabledict.immutabledict({
    networks.NetworkType.MLP: MLPConfig,
    networks.NetworkType.DISJOINT_MLPS: MLPConfig,
    networks.NetworkType.MODE_MLPS: MLPConfig,
    networks.NetworkType.CGM: CGMConfig,
})


@dataclasses.dataclass
class QLKNNModelConfig:
  """Config for QLKNNModel."""

  normalize_inputs: bool
  normalize_targets: bool
  input_names: list[str]
  target_names: list[str]
  stats_data: QLKNNStatsData | None
  flux_map: dict[str, Any]
  network_type: networks.NetworkType
  network_config: NetworkConfig

  @classmethod
  def deserialize(cls, serialized_config: bytes) -> 'QLKNNModelConfig':
    import_dict = serialization.msgpack_restore(serialized_config)
    if import_dict['stats_data'] is not None:
      import_dict['stats_data'] = QLKNNStatsData(**import_dict['stats_data'])
    import_dict['network_type'] = networks.NetworkType(
        import_dict['network_type']
    )
    import_dict['network_config'] = NETWORK_CONFIG_MAP[
        import_dict['network_type']
    ](**import_dict['network_config'])
    return cls(**import_dict)

  def serialize(self) -> bytes:
    export_dict = dataclasses.asdict(self)
    export_dict['network_type'] = export_dict['network_type'].value
    return serialization.msgpack_serialize(export_dict)


@dataclasses.dataclass
class QLKNNModelMetadata:
  """Optional metadata for QLKNNModel. Useful to identify a model."""

  path: str = ''
  name: str = ''
  version: str = ''


def _inputs_and_ranges_from_config(
    config: QLKNNModelConfig,
) -> dict[str, dict[str, float]]:
  """Returns a mapping from inputs names to min and max values."""
  inputs_and_ranges = {}
  for i, input_name in enumerate(config.input_names):
    if config.stats_data is None:
      inputs_and_ranges[input_name] = {}
    else:
      inputs_and_ranges[input_name] = {
          'min': float(config.stats_data.input_min[i]),
          'max': float(config.stats_data.input_max[i]),
      }
  return inputs_and_ranges


class QLKNNModel:
  """A JAX QLKNN Model."""

  def __init__(
      self,
      config: QLKNNModelConfig,
      params: optax.Params | None = None,
      metadata: QLKNNModelMetadata | None = None,
  ):
    self._config = config
    self._inputs_and_ranges = _inputs_and_ranges_from_config(self._config)
    self._build_network()
    self._params = params
    if metadata is None:
      metadata = QLKNNModelMetadata()
    self._metadata = metadata

  def init_params(
      self, batch_dims: tuple[int, ...], init_rng: flax_typing.PRNGKey
  ) -> None:
    self._params = jax.pmap(self._network.init)(
        init_rng,
        jnp.empty(batch_dims + (self.num_inputs,)),
    )

  @property
  def path(self) -> str:
    return self._metadata.path

  @property
  def name(self) -> str | None:
    return self._metadata.name

  @property
  def version(self) -> str:
    return self._metadata.version

  @property
  def network(self) -> nn.Module:
    return self._network

  @property
  def params(self) -> optax.Params:
    if self._params is None:
      raise ValueError('Params have not been initialized.')
    return self._params

  @params.setter
  def params(self, params: optax.Params) -> None:
    self._params = params

  @property
  def target_stats(self) -> dict[str, jax.Array]:
    if self._config.stats_data is None:
      return {}
    else:
      return {
          'mean': self._config.stats_data.target_mean,
          'stddev': self._config.stats_data.target_stddev,
      }

  @property
  def config(self) -> QLKNNModelConfig:
    return self._config

  @property
  def num_inputs(self) -> int:
    return len(self._config.input_names)

  @property
  def num_targets(self) -> int:
    return len(self._config.target_names)

  @property
  def inputs_and_ranges(self) -> dict[str, dict[str, float]]:
    return self._inputs_and_ranges

  def num_params(self) -> int:
    """Returns the number of parameters in the model."""
    return sum(p.size for p in jax.tree.leaves(self._params))

  def has_nan(self) -> bool:
    return any(
        [jnp.bool(jnp.any(jnp.isnan(p))) for p in jax.tree.leaves(self._params)]
    )

  def predict_with_params(
      self, params: optax.Params, inputs: jax.Array
  ) -> jax.Array:
    """Outputs a raw prediction.

    If targets are normalized, this will return a normalized prediction. This
    method is typically used during training while the params are changing so
    that the method can be jitted.
    Args:
      params: the model params
      inputs: model inputs

    Returns:
      the raw model prediction
    """
    if self._config.normalize_inputs and self._config.stats_data is not None:
      inputs = transforms.normalize(
          inputs,
          mean=self._config.stats_data.input_mean,
          stddev=self._config.stats_data.input_stddev,
      )
    return self._network.apply(params, inputs)

  def predict_targets(self, inputs: jax.Array) -> jax.Array:
    """Predicts the targets given the inputs.

    This will return an unnormalized prediction (e.g. fluxes in GB) whether or
    not the targets were normalized. This method is typically used at inference
    time, once the params are fixed.
    Args:
      inputs: the model input

    Returns:
      The unnormalized model predictions
    """
    outputs = self.predict_with_params(
        jax.tree_util.tree_map(lambda x: x[0], self._params), inputs
    )
    if self._config.normalize_targets and self._config.stats_data is not None:
      outputs = transforms.unnormalize(
          outputs,
          mean=self._config.stats_data.target_mean,
          stddev=self._config.stats_data.target_stddev,
      )
    return outputs

  def get_flux_from_targets(
      self, targets: jax.Array, flux_name: str
  ) -> jax.Array:
    """Compute a flux from the targets.

    Args:
      targets: An array of targets to convert to fluxes.
      flux_name: the name of the flux.

    Returns:
      A flux array.
    """
    if targets.ndim != 2:
      targets = jnp.reshape(targets, [-1, targets.shape[-1]])
    target_name = self._config.flux_map[flux_name]['target']
    denominator_name = self._config.flux_map[flux_name]['denominator']
    target_idx = self._config.target_names.index(target_name)
    if denominator_name is not None:
      denominator_idx = self._config.target_names.index(denominator_name)
      # We clip the leading flux to 0.
      flux = targets[:, target_idx] * targets[:, denominator_idx].clip(0)
    else:
      # We clip the leading flux to 0.
      flux = targets[:, target_idx].clip(0)
    return jnp.expand_dims(flux, axis=-1)

  def predict(self, inputs: jax.Array) -> dict[str, jax.Array]:
    """Predicts the fluxes given the inputs."""
    if self._config.flux_map is None:
      raise ValueError('Flux map is not specified.')
    targets = self.predict_targets(inputs)

    def _get_flux(flux_name: str) -> jax.Array:
      return self.get_flux_from_targets(
          targets=targets,
          flux_name=flux_name,
      )

    return {
        flux_name: _get_flux(flux_name)
        for flux_name in self._config.flux_map.keys()
    }

  def _build_network(self) -> None:
    """Builds the neural network."""
    assert self._config.network_config is not None
    network_kwargs = dataclasses.asdict(self._config.network_config)
    if self._config.network_type in [
        networks.NetworkType.MLP,
        networks.NetworkType.DISJOINT_MLPS,
    ]:
      network_kwargs['num_targets'] = self.num_targets
    elif self._config.network_type == networks.NetworkType.CGM:
      network_kwargs['driving_gradient_index_map'] = {
          'Ate': self._config.input_names.index('Ate'),
          'Ati': self._config.input_names.index('Ati'),
      }
    self._network = networks.NETWORK_MAP[self._config.network_type](
        **network_kwargs
    )

  def export_model(self, output_path: str, version: str | None = None) -> None:
    if version is None:
      version = VERSION
    export_dict = {
        'version': version,
        'config': self._config.serialize(),
        'params': self._params,
    }
    with open(output_path, 'wb') as f:
      f.write(serialization.msgpack_serialize(export_dict))

  @classmethod
  def load_model_from_name(
      cls,
      model_name: str,
  ) -> 'QLKNNModel':
    """Loads a QLKNNModel from a file."""
    model_path = registry.MODELS.get(model_name)
    if model_path is None:
      raise ValueError(f'Model {model_name} not found in registry.')
    return cls.load_model_from_path(model_path, model_name)

  @classmethod
  def load_model_from_path(
      cls,
      input_path: str,
      model_name: str | None = None,
  ) -> 'QLKNNModel':
    """Loads a QLKNNModel from a file."""
    logging.info('Loading QLKNNModel from %s', input_path)
    with open(input_path, 'rb') as f:
      import_dict = serialization.msgpack_restore(f.read())
    if model_name is None:
      model_name = ''
    return cls(
        config=QLKNNModelConfig.deserialize(import_dict['config']),
        params=import_dict['params'],
        metadata=QLKNNModelMetadata(
            path=input_path,
            name=model_name,
            version=import_dict['version'],
        ),
    )

  @classmethod
  def load_default_model(cls) -> 'QLKNNModel':
    """Loads the default QLKNNModel."""
    return cls.load_model_from_name(registry.DEFAULT_MODEL_NAME)

  # TODO(hamelphi): Remove this method once external dependencies are updated.
  @classmethod
  def import_model(
      cls,
      input_path: str,
  ) -> 'QLKNNModel':
    """Loads a QLKNNModel from a file."""
    logging.warning(
        'Deprecated method import_model. Use load_model_from_path instead.'
    )
    return cls.load_model_from_path(input_path)
