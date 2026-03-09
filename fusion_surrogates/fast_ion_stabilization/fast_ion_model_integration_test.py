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

"""Integration tests for registered fast ion stabilization models."""

from absl.testing import absltest
from fusion_surrogates.fast_ion_stabilization import fast_ion_model
import jax
import jax.numpy as jnp
from numpy import testing


class FastIonRegistryTest(absltest.TestCase):

  def test_load_default_model(self):
    model = (
        fast_ion_model.FastIonStabilizationModel.load_default_model_for_species(
            'H'
        )
    )
    self.assertEqual(model.species, 'H')
    self.assertEqual(model.version, '1')

  def test_load_h_model(self):
    model = (
        fast_ion_model.FastIonStabilizationModel.load_default_model_for_species(
            'H'
        )
    )
    self.assertEqual(model.species, 'H')
    raw_inputs = jnp.ones((1, fast_ion_model.NUM_INPUTS))
    output = model.predict(raw_inputs)
    self.assertEqual(output.shape, (1, fast_ion_model.NUM_OUTPUTS))

  def test_load_he3_model(self):
    model = (
        fast_ion_model.FastIonStabilizationModel.load_default_model_for_species(
            'He3'
        )
    )
    self.assertEqual(model.species, 'He3')
    raw_inputs = jnp.ones((1, fast_ion_model.NUM_INPUTS))
    output = model.predict(raw_inputs)
    self.assertEqual(output.shape, (1, fast_ion_model.NUM_OUTPUTS))

  def test_zero_fast_ions_gives_unity_h(self):
    """Physics constraint: n_fi=0 → stabilization factor = 1.0."""
    model = (
        fast_ion_model.FastIonStabilizationModel.load_default_model_for_species(
            'H'
        )
    )
    n_fi_idx = fast_ion_model.INPUT_FEATURES.index('n_fi_over_ne')
    raw_inputs = jax.random.uniform(
        jax.random.key(99),
        shape=(20, fast_ion_model.NUM_INPUTS),
        minval=jnp.array([0.1, 1.0, 0.0, 0.5, 0.0]),
        maxval=jnp.array([3.0, 5.0, 0.15, 10.0, 10.0]),
    )
    raw_inputs = raw_inputs.at[:, n_fi_idx].set(0.0)
    outputs = model.predict(raw_inputs)
    testing.assert_allclose(outputs, jnp.ones_like(outputs), atol=1e-6)


if __name__ == '__main__':
  absltest.main()
