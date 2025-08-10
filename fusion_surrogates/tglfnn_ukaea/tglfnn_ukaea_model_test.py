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

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from fusion_surrogates.tglfnn_ukaea import config, tglfnn_ukaea_model


def make_dummy_model(
    input_shape,
    n_ensemble=2,
    num_hiddens=2,
    dropout=0.1,
    normalize=True,
    unnormalize=True,
    hidden_size=5,
) -> tglfnn_ukaea_model.TGLFNNukaeaModel:
    dummy_config = config.TGLFNNukaeaModelConfig(
        n_ensemble=n_ensemble,
        num_hiddens=num_hiddens,
        dropout=dropout,
        normalize=normalize,
        unnormalize=unnormalize,
        hidden_size=hidden_size,
    )
    dummy_stats = config.TGLFNNukaeaModelStats(
        input_mean=jnp.zeros(len(config.INPUT_LABELS)),
        input_std=jnp.zeros(len(config.INPUT_LABELS)),
        output_mean=jnp.zeros(len(config.OUTPUT_LABELS)),
        output_std=jnp.zeros(len(config.OUTPUT_LABELS)),
    )

    # Instantiate a model in order to construct params
    model = tglfnn_ukaea_model.TGLFNNukaeaModel(
        config=dummy_config, stats=dummy_stats, params=None
    )
    key = jax.random.key(0)
    keys = jax.random.split(key, len(config.OUTPUT_LABELS))
    params = {
        label: model._network.init(key, jnp.ones(input_shape))
        for key, label in zip(keys, config.OUTPUT_LABELS)
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
            input_shape=(5, 10, 15),
            expected_output_shape=(5, 10, 2),
        ),
        dict(
            testcase_name="non_batched_inputs",
            input_shape=(10, 15),
            expected_output_shape=(10, 2),
        ),
        dict(
            testcase_name="single_batch_dimension",
            input_shape=(1, 3, 15),
            expected_output_shape=(1, 3, 2),
        ),
        dict(
            testcase_name="single_data_dimension",
            input_shape=(3, 1, 15),
            expected_output_shape=(3, 1, 2),
        ),
    )
    def test_predict_shape(self, input_shape, expected_output_shape):
        """Test that the predict function returns the correct shape."""
        model = make_dummy_model(input_shape)
        inputs = jnp.ones(input_shape)
        predictions = model.predict(inputs)

        for label in config.OUTPUT_LABELS:
            self.assertEqual(predictions[label].shape, expected_output_shape)


# TODO: Add test_load_params and test_predict once TGLFNNukaea open sourcing is complete

if __name__ == "__main__":
    absltest.main()
