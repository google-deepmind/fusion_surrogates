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

"""Tests for UKAEA's TGLFNN surrogate."""

import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_config
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax.numpy as jnp


class TGLFNNukaeaModelTest(parameterized.TestCase):

  def get_test_data_dir(self):
    src_dir = pathlib.Path(os.path.dirname(__file__))
    path = src_dir.joinpath("test_data")
    assert path.is_dir(), f"Path {path} is not a directory."
    return path

  def test_dummy_load(self):
    """Tests loading config, stats, and params."""
    test_data = self.get_test_data_dir()
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=tglfnn_ukaea_config.TGLFNNukaeaModelConfig.load(
            machine=tglfnn_ukaea_config.Machine("multimachine"),
            config_path=test_data / "config.yaml",
        ),
        stats=tglfnn_ukaea_config.TGLFNNukaeaModelStats.load(
            machine=tglfnn_ukaea_config.Machine("multimachine"),
            stats_path=test_data / "stats.json",
        ),
    )
    self.assertIsNone(model._params)

    # Check loading is successful
    model.load_params(
        efe_gb_pt=test_data / "dummy_torch_checkpoint.pt",
        efi_gb_pt=test_data / "dummy_torch_checkpoint.pt",
        pfi_gb_pt=test_data / "dummy_torch_checkpoint.pt",
    )
    self.assertIsNotNone(model._params)

    for key in model._config.output_labels:
      # Check output labels
      self.assertIn(key, model._params)

      # Check number of ensemble members
      self.assertLen(
          model._params.get(key).get("params"), model._config.n_ensemble
      )

      # Check number of layers
      self.assertLen(
          model._params.get(key).get("params").get("GaussianMLP_0"),
          model._config.num_hiddens,
      )

  @absltest.skipUnless(
      os.getenv("TGLFNN_UKAEA_DIR", None) is not None,
      "TGLFNN_UKAEA_DIR not set",
  )
  def test_load_params_without_errors(self):
    weights_dir = (
        pathlib.Path(os.getenv("TGLFNN_UKAEA_DIR")) / "MultiMachineHyper_1Aug25"
    )
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=tglfnn_ukaea_config.TGLFNNukaeaModelConfig.load(
            machine=tglfnn_ukaea_config.Machine("multimachine"),
            config_path=weights_dir / "config.yaml",
        ),
        stats=tglfnn_ukaea_config.TGLFNNukaeaModelStats.load(
            machine=tglfnn_ukaea_config.Machine("multimachine"),
            stats_path=weights_dir / "stats.json",
        ),
    )
    model.load_params(
        efe_gb_pt=weights_dir / "regressor_efe_gb.pt",
        efi_gb_pt=weights_dir / "regressor_efi_gb.pt",
        pfi_gb_pt=weights_dir / "regressor_pfi_gb.pt",
    )

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
    test_data = self.get_test_data_dir()
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=tglfnn_ukaea_config.TGLFNNukaeaModelConfig.load(
            machine=tglfnn_ukaea_config.Machine("multimachine"),
            config_path=test_data / "config.yaml",
        ),
        stats=tglfnn_ukaea_config.TGLFNNukaeaModelStats.load(
            machine=tglfnn_ukaea_config.Machine("multimachine"),
            stats_path=test_data / "stats.json",
        ),
    )
    model.load_params(
        efe_gb_pt=test_data / "dummy_torch_checkpoint.pt",
        efi_gb_pt=test_data / "dummy_torch_checkpoint.pt",
        pfi_gb_pt=test_data / "dummy_torch_checkpoint.pt",
    )

    inputs = jnp.ones(input_shape)
    predictions = model.predict(inputs)

    for label in tglfnn_ukaea_config.OUTPUT_LABELS:
      self.assertEqual(predictions[label].shape, expected_output_shape)


if __name__ == "__main__":
  absltest.main()
