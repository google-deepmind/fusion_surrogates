# Copyright 2024 DeepMind Technologies Limited.
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
from fusion_surrogates import qlknn_model
import jax
import jax.numpy as jnp
import numpy as np
from numpy import testing


jax.config.update("jax_enable_x64", True)


MODEL_FILENAME = "qlknn_7_11.qlknn"
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

  def setUp(self):
    super().setUp()
    model_path = os.path.join(
        os.path.dirname(__file__),
        MODEL_FILENAME,
    )
    self._model = qlknn_model.QLKNNModel.import_model(model_path)

  def test_nan(self):
    self.assertFalse(self._model.has_nan())

  @parameterized.parameters(1, 10, 100)
  def test_shape(self, batch_size):
    inputs = np.empty((1, batch_size, self._model.num_inputs))
    outputs = self._model.predict_targets(inputs)
    self.assertEqual(outputs.shape, (1, batch_size, self._model.num_targets))

  @parameterized.parameters(*_get_test_inputs_and_targets())
  def test_numerical_numpy(self, inputs, targets):
    self.assertTrue(jax.config.jax_enable_x64)
    self.assertEqual(inputs.dtype, np.float64)
    self.assertEqual(targets.dtype, np.float64)
    preds = self._model.predict_targets(inputs)
    self.assertEqual(preds.dtype, np.float64)
    testing.assert_array_almost_equal(preds, targets, decimal=10)

  @parameterized.parameters(*_get_test_inputs_and_targets())
  def test_numerical_jitted(self, inputs, targets):
    self.assertTrue(jax.config.jax_enable_x64)
    inputs = jnp.array(inputs, dtype=jnp.float64)
    targets = jnp.array(targets, dtype=jnp.float64)
    preds = jax.jit(self._model.predict_targets)(inputs)
    self.assertEqual(preds.dtype, jnp.float64)
    testing.assert_array_almost_equal(preds, targets, decimal=10)


if __name__ == "__main__":
  absltest.main()
