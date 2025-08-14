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

"""Numerical tests for QLKNN_7_11."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.qlknn import qlknn_model
from fusion_surrogates.qlknn.models import registry
import jax
import jax.numpy as jnp
import numpy as np
from numpy import testing


jax.config.update("jax_enable_x64", True)


TEST_DATA_FILENAME = "qlknn_7_11_test_data.npz"


def _get_test_inputs_and_targets():
  """Test inputs and targets for QLKNN_7_11."""
  with open(
      os.path.join(os.path.dirname(__file__), TEST_DATA_FILENAME), "rb"
  ) as f:
    test_data = np.load(f)
    inputs = test_data["inputs"]
    outputs = test_data["outputs"]
  return zip(inputs, outputs)


class Qlknn711Test(parameterized.TestCase):

  def test_nan(self):
    model = qlknn_model.QLKNNModel.load_default_model()
    self.assertFalse(model.has_nan())

  @parameterized.parameters(1, 10, 100)
  def test_shape(self, batch_size):
    model = qlknn_model.QLKNNModel.load_default_model()
    inputs = np.empty((1, batch_size, model.num_inputs))
    outputs = model.predict_targets(inputs)
    self.assertEqual(outputs.shape, (1, batch_size, model.num_targets))

  @parameterized.parameters(*_get_test_inputs_and_targets())
  def test_numerical_numpy(self, inputs, targets):
    model = qlknn_model.QLKNNModel.load_default_model()
    self.assertTrue(jax.config.jax_enable_x64)
    self.assertEqual(inputs.dtype, np.float64)
    self.assertEqual(targets.dtype, np.float64)
    preds = model.predict_targets(inputs)
    self.assertEqual(preds.dtype, np.float64)
    testing.assert_array_almost_equal(preds, targets, decimal=10)

  @parameterized.parameters(*_get_test_inputs_and_targets())
  def test_numerical_jitted(self, inputs, targets):
    model = qlknn_model.QLKNNModel.load_default_model()
    self.assertTrue(jax.config.jax_enable_x64)
    inputs = jnp.array(inputs, dtype=jnp.float64)
    targets = jnp.array(targets, dtype=jnp.float64)
    preds = jax.jit(model.predict_targets)(inputs)
    self.assertEqual(preds.dtype, jnp.float64)
    testing.assert_array_almost_equal(preds, targets, decimal=10)

  @parameterized.parameters(
      registry.MODELS.keys()
  )
  def test_registry(self, model_name):
    model = qlknn_model.QLKNNModel.load_model_from_name(model_name)
    self.assertEqual(model.name, model_name)
    self.assertEqual(model.path, registry.MODELS[model_name])
    self.assertFalse(model.has_nan())


if __name__ == "__main__":
  absltest.main()
