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

"""Tests for the ONNX QLKNN_7_11 model."""

from absl.testing import absltest
import chex
from fusion_surrogates.qlknn import qlknn_model
from fusion_surrogates.qlknn.models import registry
from jaxonnxruntime import backend
import numpy as np
import onnx


class Qlknn711OnnxTest(absltest.TestCase):

  def test_qlknn_7_11_onnx_model(self):
    """Tests that the ONNX models outputs match jax model outputs."""
    with open(
        registry.ONNX_MODELS["qlknn_7_11_v1"], "rb"
    ) as f:
      onnx_model = onnx.load(f.name)

    jax_model = qlknn_model.QLKNNModel.load_model_from_name("qlknn_7_11_v1")

    batch_size = 100
    test_input = np.random.randn(batch_size, jax_model.num_inputs).astype(
        np.float32
    )

    # Running the ONNX model using jaxonnxruntime.
    jax_model_from_onnx = backend.prepare(onnx_model)
    onnx_flat_output = jax_model_from_onnx.run([test_input])

    # Recovering the flux names from the ONNX graph
    output_names = [node.name for node in onnx_model.graph.output]

    # Reconstructing the flux dictionary.
    onnx_dict_output = dict(
        (k, v) for k, v in zip(output_names, onnx_flat_output)
    )

    # Running the original JAX model.
    jax_output = jax_model.predict(test_input)

    # Checking that the output names match the expected output keys.
    self.assertEmpty(set(onnx_dict_output.keys()) ^ set(jax_output.keys()))
    # Checking that the output values match.
    chex.assert_trees_all_close(onnx_dict_output, jax_output, atol=1e-06)


if __name__ == "__main__":
  absltest.main()
