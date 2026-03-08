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

import tempfile

from absl.testing import absltest
from fusion_surrogates.fast_ion_stabilization import fast_ion_model
import jax
import jax.numpy as jnp
import numpy as np
from numpy import testing


def _make_test_model(
    seed: int = 0,
) -> fast_ion_model.FastIonStabilizationModel:
  """Creates a randomly-initialized model for testing."""
  input_stats = fast_ion_model.InputStats(
      mean=np.array([0.5, 2.0, 0.05, 1.5, 3.0], dtype=np.float32),
      std=np.array([0.3, 1.0, 0.03, 0.5, 2.0], dtype=np.float32),
  )
  config = fast_ion_model.FastIonStabilizationModelConfig(
      num_hiddens=2,
      hidden_size=8,
      activation='tanh',
      input_stats=input_stats,
  )
  model = fast_ion_model.FastIonStabilizationModel(
      config=config, species='D', version='2'
  )
  rng = jax.random.PRNGKey(seed)
  params = model.network.init(rng, jnp.ones((1, fast_ion_model.NUM_INPUTS)))
  model.params = params
  return model


class FastIonStabilizationModelTest(absltest.TestCase):

  def test_predict_output_shape(self):
    model = _make_test_model()
    batch_size = 10
    raw_inputs = jax.random.uniform(
        jax.random.key(1),
        shape=(batch_size, fast_ion_model.NUM_INPUTS),
    )
    outputs = model.predict(raw_inputs)
    self.assertEqual(outputs.shape, (batch_size, fast_ion_model.NUM_OUTPUTS))

  def test_predict_output_dtype(self):
    model = _make_test_model()
    raw_inputs = jnp.ones((1, fast_ion_model.NUM_INPUTS), dtype=jnp.float32)
    outputs = model.predict(raw_inputs)
    self.assertEqual(outputs.dtype, jnp.float32)

  def test_predict_batched(self):
    model = _make_test_model()
    raw_inputs = jax.random.uniform(
        jax.random.key(2),
        shape=(3, 5, fast_ion_model.NUM_INPUTS),
    )
    outputs = model.predict(raw_inputs)
    self.assertEqual(outputs.shape, (3, 5, fast_ion_model.NUM_OUTPUTS))

  def test_zero_fast_ions_gives_unity(self):
    """When n_fi/ne = 0, the stabilization factor must be exactly 1.0."""
    model = _make_test_model()
    n_fi_idx = fast_ion_model.INPUT_FEATURES.index('n_fi_over_ne')
    raw_inputs = jax.random.uniform(
        jax.random.key(3),
        shape=(20, fast_ion_model.NUM_INPUTS),
    )
    raw_inputs = raw_inputs.at[:, n_fi_idx].set(0.0)
    outputs = model.predict(raw_inputs)
    testing.assert_allclose(np.array(outputs), np.ones_like(outputs), atol=1e-6)

  def test_predict_without_params_raises(self):
    config = fast_ion_model.FastIonStabilizationModelConfig(
        num_hiddens=2,
        hidden_size=8,
        activation='tanh',
        input_stats=fast_ion_model.InputStats(
            mean=np.zeros(fast_ion_model.NUM_INPUTS, dtype=np.float32),
            std=np.ones(fast_ion_model.NUM_INPUTS, dtype=np.float32),
        ),
    )
    model = fast_ion_model.FastIonStabilizationModel(config=config)
    with self.assertRaises(ValueError):
      model.predict(jnp.ones((1, fast_ion_model.NUM_INPUTS)))

  def test_config_serialize_deserialize(self):
    input_stats = fast_ion_model.InputStats(
        mean=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        std=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    )
    config = fast_ion_model.FastIonStabilizationModelConfig(
        num_hiddens=3,
        hidden_size=16,
        activation='relu',
        input_stats=input_stats,
    )
    serialized = config.serialize()
    restored = fast_ion_model.FastIonStabilizationModelConfig.deserialize(
        serialized
    )
    self.assertEqual(restored.num_hiddens, config.num_hiddens)
    self.assertEqual(restored.hidden_size, config.hidden_size)
    self.assertEqual(restored.activation, config.activation)
    testing.assert_array_equal(
        np.array(restored.input_stats.mean),
        np.array(config.input_stats.mean),
    )
    testing.assert_array_equal(
        np.array(restored.input_stats.std),
        np.array(config.input_stats.std),
    )

  def test_deterministic_predictions(self):
    model = _make_test_model(seed=42)
    raw_inputs = jax.random.uniform(
        jax.random.key(5),
        shape=(5, fast_ion_model.NUM_INPUTS),
    )
    outputs_1 = model.predict(raw_inputs)
    outputs_2 = model.predict(raw_inputs)
    testing.assert_array_equal(np.array(outputs_1), np.array(outputs_2))

  def test_export_import_round_trip(self):
    model = _make_test_model()
    raw_inputs = jax.random.uniform(
        jax.random.key(6),
        shape=(5, fast_ion_model.NUM_INPUTS),
    )
    original_outputs = model.predict(raw_inputs)
    with tempfile.NamedTemporaryFile(suffix='.fistab') as f:
      model.export_model(f.name)
      loaded = fast_ion_model.FastIonStabilizationModel.load_model_from_path(
          f.name, 'test_model'
      )
    loaded_outputs = loaded.predict(raw_inputs)
    testing.assert_array_equal(
        np.array(original_outputs), np.array(loaded_outputs)
    )
    self.assertEqual(loaded.name, 'test_model')
    self.assertEqual(loaded.species, 'D')
    self.assertEqual(loaded.version, '2')
    testing.assert_array_equal(
        np.array(loaded.config.input_stats.mean),
        np.array(model.config.input_stats.mean),
    )


if __name__ == '__main__':
  absltest.main()
