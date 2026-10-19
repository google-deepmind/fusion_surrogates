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
import tglfnn_ukaea as tglfnn_ukaea_lib


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
        "version": np.array("3.2.1"),
        "input_labels": np.array(["input_a", "input_b"]),
        "output_labels": np.array(["flux_a", "flux_b"]),
        "input_mean": np.zeros((2, 2)),
        "input_std": np.ones((2, 2)),
        "output_mean": np.array([-10.0, 5.0]),
        "output_std": np.array([3.0, 0.0]),
        "variance_epsilon": np.zeros(2, dtype=np.float32),
        "num_estimators": np.array([2, 2]),
        "num_layers": np.array([1, 1]),
        "hidden_size": np.array([2, 2]),
        "activation": np.array("relu"),
    }
    for label in ("flux_a", "flux_b"):
      model_dict[f"{label}/layer_0/weight"] = np.zeros((2, 2, 2))
      model_dict[f"{label}/layer_0/bias"] = np.array([[1.0, 0.0], [3.0, 0.0]])
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

  @parameterized.parameters("multimachine", "step")
  def test_matches_onnx(self, machine):
    """The model reproduces the original ONNX models."""
    # pylint: disable=protected-access
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(machine=machine)
    # Reference outputs of the ONNX models, distributed with the weights.
    reference = tglfnn_ukaea_lib.load_reference(machine, model.version)
    predictions = model.predict(jnp.asarray(reference["inputs"]))
    for i, label in enumerate(model.output_labels):
      expected = reference[label]
      output_std = model._output_stds[i]
      np.testing.assert_allclose(
          predictions[label][..., 0],
          expected[..., 0],
          rtol=0,
          atol=1e-4 * output_std,
      )
      # Where the ensemble members agree, the ONNX models' variance is only
      # defined to float32 rounding (they compute it as E[x^2] - E[x]^2).
      np.testing.assert_allclose(
          predictions[label][..., 1],
          expected[..., 1],
          rtol=1e-2,
          atol=1e-6 * output_std**2,
      )


if __name__ == "__main__":
  absltest.main()
