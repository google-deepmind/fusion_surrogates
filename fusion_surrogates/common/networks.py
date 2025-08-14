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

"""Network architectures for surrogate models."""

from collections.abc import Callable, Mapping
import enum
from typing import Final

import flax.linen as nn
import immutabledict
import jax
import jax.numpy as jnp


_ACTIVATION_FNS: Final[Mapping[str, Callable[[jax.Array], jax.Array]]] = (
    immutabledict.immutabledict({
        'relu': nn.relu,
        'tanh': nn.tanh,
        'sigmoid': nn.sigmoid,
    })
)


class MLP(nn.Module):
  """Basic Fully Connected Network."""

  num_hiddens: int
  hidden_size: int
  num_targets: int | None
  activation: str

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    for _ in range(self.num_hiddens):
      x = _ACTIVATION_FNS[self.activation](nn.Dense(self.hidden_size)(x))
    if self.num_targets is not None:
      x = nn.Dense(self.num_targets)(x)
    return x


class GaussianMLP(nn.Module):
  """An MLP with dropout, outputting a mean and variance."""

  num_hiddens: int
  hidden_size: int
  dropout: float
  activation: str

  @nn.compact
  def __call__(
      self,
      x,
      deterministic: bool = False,
  ):
    for _ in range(self.num_hiddens - 1):
      x = nn.Dense(self.hidden_size)(x)
      x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
      x = _ACTIVATION_FNS[self.activation](x)
    mean_and_var = nn.Dense(2)(x)
    mean = mean_and_var[..., 0]
    var = mean_and_var[..., 1]
    var = nn.softplus(var)

    return jnp.stack([mean, var], axis=-1)


class GaussianMLPEnsemble(nn.Module):
  """An ensemble of GaussianMLPs."""

  n_ensemble: int
  num_hiddens: int
  hidden_size: int
  dropout: float
  activation: str

  @nn.compact
  def __call__(
      self,
      x,
      deterministic: bool = False,
  ):
    ensemble_output = jnp.stack(
        [
            GaussianMLP(
                self.num_hiddens,
                self.hidden_size,
                self.dropout,
                self.activation,
            )(x, deterministic=deterministic)
            for _ in range(self.n_ensemble)
        ],
        axis=0,
    )
    mean = jnp.mean(ensemble_output[..., 0], axis=0)
    aleatoric = jnp.mean(ensemble_output[..., 1], axis=0)
    epistemic = jnp.var(ensemble_output[..., 0], axis=0)
    return jnp.stack([mean, aleatoric + epistemic], axis=-1)


class DisjointMLPs(nn.Module):
  """Disjoint MLPs, one per target."""

  num_hiddens: int
  hidden_size: int
  num_targets: int
  activation: str

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [
            MLP(
                num_hiddens=self.num_hiddens,
                hidden_size=self.hidden_size,
                num_targets=1,
                activation=self.activation,
            )(x)
            for _ in range(self.num_targets)
        ],
        axis=-1,
    )


class ModeMLPs(nn.Module):
  """Independent MLP for each mode."""

  num_hiddens: int
  hidden_size: int
  activation: str

  def _mlp(self, num_targets: int) -> nn.Module:
    return MLP(
        num_hiddens=self.num_hiddens,
        hidden_size=self.hidden_size,
        num_targets=num_targets,
        activation=self.activation,
    )

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    outputs = []
    # ITG
    outputs.append(self._mlp(3)(x))
    # TEM
    outputs.append(self._mlp(3)(x))
    # ETG
    outputs.append(self._mlp(1)(x))
    # Gamma max
    outputs.append(self._mlp(1)(x))
    return jnp.concatenate(outputs, axis=-1)


class _CGMNetModule(nn.Module):
  """Network architecture based on the Critical Gradient Model.

  Implementation of CGMNets as described in:
  Inclusion of Physics Constraints in Neural Network Surrogate Models for
  Fusion Simulation
  Horn, P. (Author). 28 May 2020
  """

  torso_num_hiddens: int
  torso_hidden_size: int
  head_num_hiddens: int
  head_hidden_size: int
  activation: str
  critical_gradient_index: int
  mode: str

  def flux_names(self) -> list[str]:
    return [f'{flux}{self.mode}' for flux in ['efe', 'efi', 'pfe']]

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    xg = jnp.delete(x, self.critical_gradient_index, axis=-1)
    xs = jnp.expand_dims(x[..., self.critical_gradient_index], axis=-1)
    # Shared part of the Neural network.
    torso = MLP(
        num_hiddens=self.torso_num_hiddens,
        hidden_size=self.torso_hidden_size,
        num_targets=None,
        activation=self.activation,
        name='shared_torso',
    )(xg)

    def c(x, name: str):
      return MLP(
          num_hiddens=self.head_num_hiddens,
          hidden_size=self.head_hidden_size,
          num_targets=1,
          activation=self.activation,
          name=name,
      )(x)

    outputs = []

    c1 = c(torso, 'c1_head')
    # efe
    # This relu clips the c2 exponent to a positive value.
    c2e = nn.relu(c(torso, 'c2e_head'))
    c3e = c(torso, 'c3e_head')
    outputs.append(c3e * nn.relu(xs - c1) ** (c2e + 1))
    if self.mode != 'ETG':
      # efi
      c2i = nn.relu(c(torso, 'c2i_head'))
      c3i = c(torso, 'c3i_head')
      outputs.append(c3i * nn.relu(xs - c1) ** (c2i + 1))
      # pfe
      fd = c(jnp.concatenate([xs, torso], axis=-1), 'fd_head')
      outputs.append(fd * nn.relu(xs - c1))

    return jnp.concatenate(outputs, axis=-1)


# Driving gradient for each mode.
_DRIVING_GRADIENTS_MODE_MAP = immutabledict.immutabledict(
    {'ITG': 'Ati', 'TEM': 'Ate', 'ETG': 'Ate'}
)


class CGMNets(nn.Module):
  """Critical Gradient Model Networks for ITG, TEM and ETG modes."""

  torso_num_hiddens: int
  torso_hidden_size: int
  head_num_hiddens: int
  head_hidden_size: int
  activation: str
  driving_gradient_index_map: Mapping[str, int]

  def flux_names(self) -> list[str]:
    flux_names = []
    for mode in _DRIVING_GRADIENTS_MODE_MAP:
      fluxes = ['efe', 'efi', 'pfe'] if mode != 'ETG' else ['efe']
      flux_names.extend([f'{flux}{mode}' for flux in fluxes])
    flux_names.append('gamma_max')
    return flux_names

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    outputs = []
    for mode, driving_gradient_name in _DRIVING_GRADIENTS_MODE_MAP.items():
      outputs.append(
          _CGMNetModule(
              torso_num_hiddens=self.torso_num_hiddens,
              torso_hidden_size=self.torso_hidden_size,
              head_num_hiddens=self.head_num_hiddens,
              head_hidden_size=self.head_hidden_size,
              activation=self.activation,
              critical_gradient_index=self.driving_gradient_index_map[
                  driving_gradient_name
              ],
              mode=mode,
              name=f'{mode}_cgm',
          )(x)
      )
    # Gamma max.
    outputs.append(
        MLP(
            num_hiddens=self.torso_num_hiddens,
            hidden_size=self.torso_hidden_size,
            num_targets=1,
            activation=self.activation,
            name='gamma_max_mlp',
        )(x)
    )
    return jnp.concatenate(outputs, axis=-1)


class NetworkType(enum.Enum):
  MLP = 'mlp'
  GAUSSIAN_MLP = 'gaussian_mlp'
  DISJOINT_MLPS = 'disjoint_mlps'
  MODE_MLPS = 'mode_mlps'
  GAUSSIAN_MLP_ENSEMBLE = 'gaussian_mlp_ensemble'
  CGM = 'cgm'


NETWORK_MAP: Final[Mapping[NetworkType, type[nn.Module]]] = (
    immutabledict.immutabledict({
        NetworkType.MLP: MLP,
        NetworkType.GAUSSIAN_MLP: GaussianMLP,
        NetworkType.DISJOINT_MLPS: DisjointMLPs,
        NetworkType.MODE_MLPS: ModeMLPs,
        NetworkType.GAUSSIAN_MLP_ENSEMBLE: GaussianMLPEnsemble,
        NetworkType.CGM: CGMNets,
    })
)
