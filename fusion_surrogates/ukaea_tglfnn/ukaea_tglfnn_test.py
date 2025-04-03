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

import jax.numpy as jnp
from absl.testing import absltest
from numpy import testing

from fusion_surrogates.ukaea_tglfnn import pytorch_model
from fusion_surrogates.ukaea_tglfnn import onnx_model


class PyTorchTGLFNNModelTest(absltest.TestCase):
    model = pytorch_model.PytorchTGLFNNModel(
        config_path="models/1.0.1/config.yaml",
        stats_path="models/1.0.1/stats.json",
        efe_gb_checkpoint_path="models/1.0.1/regressor_efe_gb.pt",
        efi_gb_checkpoint_path="models/1.0.1/regressor_efi_gb.pt",
        pfi_gb_checkpoint_path="models/1.0.1/regressor_pfi_gb.pt",
        map_location="cpu",
    )

    reference_inputs = jnp.load("test_data/input.npy")
    reference_outputs = jnp.load("test_data/output.npy")

    def test_matches_reference(self):
        predicted_outputs = self.model.predict(self.reference_inputs)
        testing.assert_allclose(
            self.reference_outputs[..., 0], predicted_outputs, rtol=1e-3
        )


class ONNXTGLFNNModelTest(absltest.TestCase):
    model = onnx_model.ONNXTGLFNNModel(
        efe_onnx_path="models/1.0.1/regressor_efe_gb_onnx.onnx",
        efi_onnx_path="models/1.0.1/regressor_efi_gb_onnx.onnx",
        pfi_onnx_path="models/1.0.1/regressor_pfi_gb_onnx.onnx",
    )

    reference_inputs = jnp.load("test_data/input.npy")
    reference_outputs = jnp.load("test_data/output.npy")

    def test_matches_reference(self):
        predicted_outputs = self.model.predict(self.reference_inputs)
        testing.assert_allclose(
            self.reference_outputs, predicted_outputs[..., 0], rtol=1e-3
        )


if __name__ == "__main__":
    absltest.main()
