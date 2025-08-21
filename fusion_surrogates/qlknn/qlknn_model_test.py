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

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.common import networks
from fusion_surrogates.qlknn import qlknn_model
from fusion_surrogates.qlknn import qlknn_model_test_utils
import jax
import jax.numpy as jnp
from numpy import testing


class QlknnModelTest(parameterized.TestCase):
  """Tests for qlknn_model."""

  def test_predict_targets_output_shape(self):
    """Tests model output shape."""
    config = qlknn_model_test_utils.get_test_model_config()
    # Batches are 2d where the first dimension is the number of devices.
    batch_dims = (1, 10)
    model = qlknn_model_test_utils.init_model(config, batch_dims)
    inputs = jnp.empty(batch_dims + (model.num_inputs,))
    outputs = model.predict_targets(inputs)
    self.assertEqual(outputs.shape, batch_dims + (model.num_targets,))

  def test_model_export_import(self):
    """Tests export_model and import_model."""
    config = qlknn_model_test_utils.get_test_model_config()
    batch_dims = (1, 10)
    orig_model = qlknn_model_test_utils.init_model(config, batch_dims)
    with tempfile.NamedTemporaryFile() as f:
      orig_model.export_model(f.name, 'my_version')
      imported_model = qlknn_model.QLKNNModel.load_model_from_path(f.name)
    inputs = jnp.empty(batch_dims + (orig_model.num_inputs,))
    orig_outputs = orig_model.predict_targets(inputs)
    imported_outputs = imported_model.predict_targets(inputs)
    testing.assert_array_equal(orig_outputs, imported_outputs)
    self.assertEqual(imported_model.version, 'my_version')

  def test_predict(self):
    """Tests that predict outputs are computed from targets as expected."""
    config = qlknn_model_test_utils.get_test_model_config()
    batch_dims = (1, 10)
    model = qlknn_model_test_utils.init_model(config, batch_dims)
    inputs = jax.random.uniform(
        jax.random.key(0), shape=batch_dims + (model.num_inputs,)
    )
    targets = model.predict_targets(inputs)
    # Flattening the batch dims.
    targets = targets.reshape(-1, model.num_targets)
    fluxes = model.predict(inputs)

    def _get_column(idx):
      """Get a (B, 1) column from targets."""
      return targets[:, idx][:, None]

    testing.assert_array_equal(fluxes['flux0'], _get_column(0).clip(0))
    testing.assert_array_equal(
        fluxes['flux1'], _get_column(1) * _get_column(0).clip(0)
    )
    testing.assert_array_equal(
        fluxes['flux2'], _get_column(2) * _get_column(0).clip(0)
    )

  def test_get_fluxes_from_targets(self):
    """Tests that get_fluxes_from_targets outputs are computed as expected."""
    config = qlknn_model_test_utils.get_test_model_config()
    model = qlknn_model.QLKNNModel(config=config)
    targets = jax.random.uniform(
        jax.random.key(0), shape=(1, 10, model.num_targets)
    )
    targets = targets.reshape(-1, model.num_targets)

    def _get_column(idx):
      """Get a (B, 1) column from targets."""
      return targets[:, idx][:, None]

    testing.assert_array_equal(
        model.get_flux_from_targets(targets, 'flux0'),
        _get_column(0).clip(0),
    )
    testing.assert_array_equal(
        model.get_flux_from_targets(targets, 'flux1'),
        _get_column(1) * _get_column(0).clip(0),
    )
    testing.assert_array_equal(
        model.get_flux_from_targets(targets, 'flux2'),
        _get_column(2) * _get_column(0).clip(0),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='mlp',
          network_type=networks.NetworkType.MLP,
          network_class=networks.MLP,
          network_config=qlknn_model.MLPConfig(
              num_hiddens=2,
              hidden_size=10,
              activation='relu',
          ),
      ),
      dict(
          testcase_name='disjoint_mlps',
          network_type=networks.NetworkType.DISJOINT_MLPS,
          network_class=networks.DisjointMLPs,
          network_config=qlknn_model.MLPConfig(
              num_hiddens=3,
              hidden_size=11,
              activation='tanh',
          ),
      ),
      dict(
          testcase_name='mode_mlps',
          network_type=networks.NetworkType.MODE_MLPS,
          network_class=networks.ModeMLPs,
          network_config=qlknn_model.MLPConfig(
              num_hiddens=4,
              hidden_size=9,
              activation='sigmoid',
          ),
      ),
      dict(
          testcase_name='cgm',
          network_type=networks.NetworkType.CGM,
          network_class=networks.CGMNets,
          network_config=qlknn_model.CGMConfig(
              torso_num_hiddens=2,
              torso_hidden_size=10,
              head_num_hiddens=3,
              head_hidden_size=11,
              activation='relu',
          ),
      ),
  )
  def test_build_mlp(self, network_type, network_class, network_config):
    config = qlknn_model_test_utils.get_test_model_config()
    config.network_type = network_type
    config.network_config = network_config
    model = qlknn_model.QLKNNModel(config=config)
    self.assertIsInstance(model.network, network_class)

  def test_registry(self):
    model = qlknn_model.QLKNNModel.load_model_from_name('qlknn_7_11_v1')
    self.assertEqual(model.version, '11D')

  def test_default_model(self):
    model = qlknn_model.QLKNNModel.load_default_model()
    self.assertEqual(model.version, '11D')

if __name__ == '__main__':
  absltest.main()
