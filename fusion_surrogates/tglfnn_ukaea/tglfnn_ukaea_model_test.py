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

"""Tests for UKAEA's TGLFNN surrogate."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax
import jax.numpy as jnp
import numpy as np


class TGLFNNukaeaModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="batched_inputs",
          input_shape=(5, 10, 13),
          expected_output_shape=(5, 10, 2),
      ),
      dict(
          testcase_name="non_batched_inputs",
          input_shape=(10, 13),
          expected_output_shape=(10, 2),
      ),
      dict(
          testcase_name="single_batch_dimension",
          input_shape=(1, 3, 13),
          expected_output_shape=(1, 3, 2),
      ),
      dict(
          testcase_name="single_data_dimension",
          input_shape=(3, 1, 13),
          expected_output_shape=(3, 1, 2),
      ),
  )
  def test_predict_shape(self, input_shape, expected_output_shape):
    """Test that the predict function returns the correct shape."""
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(machine="multimachine")
    inputs = jnp.ones(input_shape)
    predictions = model.predict(inputs)

    for label in model.output_labels:
      self.assertEqual(predictions[label].shape, expected_output_shape)

  @parameterized.product(
      input_shape=((2,), (4, 2), (2, 3, 2)),
      input_dtype=(jnp.float32, jnp.float64),
      use_jit=(False, True),
  )
  def test_predict_rescales_mean_and_variance(
      self, input_shape, input_dtype, use_jit
  ):
    # Two equally weighted members have means 1 and 3, variance log(2).
    model_dict = {
        "input_labels": ["input_a", "input_b"],
        "config": {"num_estimators": 2, "model_size": 1},
        "params": {
            label: {
                f"MLP_{i}": {
                    "FullyConnectedLayer_0": {
                        "weight": np.zeros((2, 2)),
                        "bias": np.array([mean, 0.0]),
                    }
                }
                for i, mean in enumerate((1.0, 3.0))
            }
            for label in ("flux_a", "flux_b")
        },
        "stats": {
            "input_a": {"mean": 0.0, "std": 1.0},
            "input_b": {"mean": 0.0, "std": 1.0},
            "flux_a": {"mean": -10.0, "std": 3.0},
            "flux_b": {"mean": 5.0, "std": 0.0},
        },
    }
    self.addCleanup(jax.config.update, "jax_enable_x64", jax.config.x64_enabled)
    jax.config.update("jax_enable_x64", True)
    with mock.patch.object(
        tglfnn_ukaea_model.tglfnn_ukaea_lib,
        "load",
        return_value=model_dict,
    ):
      model = tglfnn_ukaea_model.TGLFNNukaeaModel()
    predict = jax.jit(model.predict) if use_jit else model.predict
    predictions = predict(jnp.ones(input_shape, dtype=input_dtype))
    for label, offset, scale in (("flux_a", -10.0, 3.0), ("flux_b", 5.0, 1.0)):
      result = predictions[label]
      self.assertEqual(result.shape, input_shape[:-1] + (2,))
      self.assertEqual(result.dtype, input_dtype)
      np.testing.assert_allclose(result[..., 0], 2.0 * scale + offset)
      np.testing.assert_allclose(
          result[..., 1], (1.0 + np.log(2.0)) * scale**2, rtol=1e-6
      )
      self.assertTrue(np.all(result[..., 1] >= 0.0))


if __name__ == "__main__":
  absltest.main()
