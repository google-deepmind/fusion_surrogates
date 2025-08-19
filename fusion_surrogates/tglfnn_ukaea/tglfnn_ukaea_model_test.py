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

"""Tests for UKAEA's TGLFNN surrogate"""

import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import requests
from absl.testing import absltest, parameterized

from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_config
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model


def make_dummy_model(
    input_shape,
    n_ensemble=2,
    num_hiddens=2,
    dropout=0.1,
    hidden_size=5,
) -> tglfnn_ukaea_model.TGLFNNukaeaModel:
    dummy_config = tglfnn_ukaea_config.TGLFNNukaeaModelConfig(
        n_ensemble=n_ensemble,
        num_hiddens=num_hiddens,
        dropout=dropout,
        hidden_size=hidden_size,
        machine="multimachine",
    )
    dummy_stats = tglfnn_ukaea_config.TGLFNNukaeaModelStats(
        input_mean=jnp.zeros(len(dummy_config.input_labels)),
        input_std=jnp.zeros(len(dummy_config.input_labels)),
        output_mean=jnp.zeros(len(dummy_config.output_labels)),
        output_std=jnp.zeros(len(dummy_config.output_labels)),
    )

    # Instantiate a model in order to construct params
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=dummy_config, stats=dummy_stats, params=None
    )
    key = jax.random.key(0)
    keys = jax.random.split(key, len(dummy_config.output_labels))
    params = {
        label: model._network.init(key, jnp.ones(input_shape))
        for key, label in zip(keys, dummy_config.output_labels)
    }

    # Recreate the model with the given params
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=dummy_config, stats=dummy_stats, params=params
    )

    return model


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
        model = make_dummy_model(input_shape)
        inputs = jnp.ones(input_shape)
        predictions = model.predict(inputs)

        for label in tglfnn_ukaea_config.OUTPUT_LABELS:
            self.assertEqual(predictions[label].shape, expected_output_shape)

    def test_load(self):
        """Tests loading config, stats, and params from the Github repo."""

        def download(src, dest):
            file_name = os.path.basename(src)

            response = requests.get(src, stream=True)
            response.raise_for_status()

            with open(dest / file_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return dest / file_name

        efe_weights_url = "https://raw.githubusercontent.com/ukaea/tglfnn-ukaea/main/MultiMachineHyper_1Aug25/regressor_efe_gb.pt"
        config_url = "https://raw.githubusercontent.com/ukaea/tglfnn-ukaea/main/MultiMachineHyper_1Aug25/config.yaml"
        stats_url = "https://raw.githubusercontent.com/ukaea/tglfnn-ukaea/main/MultiMachineHyper_1Aug25/stats.json"

        test_dir = Path(tempfile.mkdtemp())
        efe_weights_path = download(efe_weights_url, test_dir)
        config_path = download(config_url, test_dir)
        stats_path = download(stats_url, test_dir)

        model = tglfnn_ukaea_model.TGLFNNukaeaModel(
            config=tglfnn_ukaea_config.TGLFNNukaeaModelConfig.load(
                machine="multimachine", config_path=config_path
            ),
            stats=tglfnn_ukaea_config.TGLFNNukaeaModelStats.load(
                machine="multimachine", stats_path=stats_path
            ),
        )
        assert model._params is None

        # Check loading is successful
        model.load_params(
            efe_gb_pt=efe_weights_path,
            efi_gb_pt=efe_weights_path,
            pfi_gb_pt=efe_weights_path,
        )
        assert model._params is not None

        for key in model._config.output_labels:
            # Check output labels
            assert key in model._params

            # Check number of ensemble members
            assert len(model._params.get(key).get("params")) == model._config.n_ensemble

            # Check number of layers
            assert (
                len(model._params.get(key).get("params").get("GaussianMLP_0"))
                == model._config.num_hiddens
            )


if __name__ == "__main__":
    absltest.main()
