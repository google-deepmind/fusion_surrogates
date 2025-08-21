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

from absl.testing import absltest
import flax.linen as nn
from fusion_surrogates.common import networks
import jax
import jax.numpy as jnp


class NetworksTest(absltest.TestCase):

  def _test_network_io(
      self, network: nn.Module, num_inputs: int, num_targets: int, **kwargs
  ):
    """Checks that the network inputs and outputs have the correct shapes."""
    # Initialize the parameters
    batch_size = 3

    key = jax.random.PRNGKey(0)

    params = network.init(key, jnp.zeros((batch_size, num_inputs)))

    # Apply the network to some input data
    inputs = jnp.ones((batch_size, num_inputs))
    outputs = network.apply(params, inputs, **kwargs)
    # Check the shape of the outputs
    self.assertEqual(outputs.shape, (batch_size, num_targets))

  def test_mlp(self):
    num_inputs = 5
    num_targets = 5

    network = networks.NETWORK_MAP[networks.NetworkType.MLP](
        num_hiddens=2,
        hidden_size=7,
        num_targets=num_targets,
        activation='relu',
    )
    self._test_network_io(network, num_inputs, num_targets)

  def test_mlp_no_targets(self):
    num_inputs = 5
    hidden_size = 7

    network = networks.NETWORK_MAP[networks.NetworkType.MLP](
        num_hiddens=2,
        hidden_size=hidden_size,
        num_targets=None,
        activation='relu',
    )
    self._test_network_io(network, num_inputs, hidden_size)

  def test_gaussian_mlp(self):
    num_inputs = 5
    num_targets = 2  # Gaussian MLPs always have 2 outputs.

    network = networks.NETWORK_MAP[networks.NetworkType.GAUSSIAN_MLP](
        num_hiddens=2,
        hidden_size=7,
        dropout=0.1,
        activation='relu',
    )
    self._test_network_io(network, num_inputs, num_targets, deterministic=True)

  def test_gaussian_mlp_ensemble(self):
    num_inputs = 5
    num_targets = 2  # Gaussian MLPs always have 2 outputs.
    n_ensemble = 3

    network = networks.NETWORK_MAP[networks.NetworkType.GAUSSIAN_MLP_ENSEMBLE](
        num_hiddens=2,
        hidden_size=7,
        dropout=0.1,
        activation='relu',
        n_ensemble=n_ensemble,
    )

    self._test_network_io(network, num_inputs, num_targets, deterministic=True)

  def test_disjoint_mlps(self):
    num_inputs = 5
    num_targets = 5
    network = networks.NETWORK_MAP[networks.NetworkType.DISJOINT_MLPS](
        num_hiddens=2,
        hidden_size=7,
        num_targets=num_targets,
        activation='relu',
    )
    self._test_network_io(network, num_inputs, num_targets)

  def test_mode_mlps(self):
    num_inputs = 5
    num_targets = 8  # Mode MLPs always have 8 outputs.
    network = networks.NETWORK_MAP[networks.NetworkType.MODE_MLPS](
        num_hiddens=2,
        hidden_size=7,
        activation='relu',
    )
    self._test_network_io(network, num_inputs, num_targets)

  def test_cgm(self):
    num_inputs = 5
    num_targets = 8  # CGMNets always have 8 outputs.
    network = networks.NETWORK_MAP[networks.NetworkType.CGM](
        torso_num_hiddens=2,
        torso_hidden_size=7,
        head_num_hiddens=3,
        head_hidden_size=8,
        activation='relu',
        driving_gradient_index_map={
            'Ati': 0,
            'Ate': 1,
        },
    )
    self._test_network_io(network, num_inputs, num_targets)


if __name__ == '__main__':
  absltest.main()
