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
from absl.testing import parameterized
from fusion_surrogates.fast_ion_stabilization import fast_ion_model
import jax
import jax.numpy as jnp
import numpy as np
from numpy import testing

_SMAG = 1.0
_Q = 2.0
_N_FI_OVER_NE = 0.05
_RLTI_FI = 30.0
_T_FI_OVER_TE_VALUES = np.array([1.0, 2.0, 5.0, 10.0, 15.0], dtype=np.float32)

_EXPECTED_H_OUTPUTS = np.array(
    [0.7342398, 0.9100813, 1.1069683, 1.0865688, 1.0558544],
    dtype=np.float32,
)

_EXPECTED_HE3_OUTPUTS = np.array(
    [0.1983749, 0.4569806, 0.9586841, 1.2051169, 1.2053574],
    dtype=np.float32,
)


class FastIonRegistryTest(parameterized.TestCase):

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

  @parameterized.named_parameters(
      dict(
          testcase_name='H',
          model_name='fast_ion_H_v1',
          expected=_EXPECTED_H_OUTPUTS,
      ),
      dict(
          testcase_name='He3',
          model_name='fast_ion_He3_v1',
          expected=_EXPECTED_HE3_OUTPUTS,
      ),
  )
  def test_regression_sweep(self, model_name: str, expected: np.ndarray):
    model = fast_ion_model.FastIonStabilizationModel.load_model_from_name(
        model_name
    )
    raw_inputs = jnp.stack([
        jnp.array([_SMAG, _Q, _N_FI_OVER_NE, t, _RLTI_FI])
        for t in _T_FI_OVER_TE_VALUES
    ])
    outputs = model.predict(raw_inputs)
    testing.assert_allclose(
        np.array(outputs).flatten(),
        expected,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()
