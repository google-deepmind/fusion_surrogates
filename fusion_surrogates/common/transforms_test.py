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

from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.common import transforms
import jax
import jax.numpy as jnp
from numpy import testing


class TransformsTest(parameterized.TestCase):
  """Tests for transforms."""

  def test_normalize(self):
    num_features = 20
    batch_shape = (1, 10)
    data_shape = batch_shape + (num_features,)
    data = jax.random.uniform(jax.random.key(0), shape=data_shape)
    data_mean = data.mean(axis=(0, 1))
    data_std = data.std(axis=(0, 1))
    normalized_data = transforms.normalize(
        data, mean=data_mean, stddev=data_std
    )
    testing.assert_array_almost_equal(
        normalized_data.mean(axis=(0, 1)), jnp.zeros(num_features)
    )
    testing.assert_array_almost_equal(
        normalized_data.std(axis=(0, 1)), jnp.ones(num_features)
    )

  def test_unnormalize(self):
    data_shape = (2, 25, 5)
    data = jax.random.uniform(jax.random.key(0), shape=data_shape)
    data_mean = data.mean(axis=(0, 1))
    data_std = data.std(axis=(0, 1))
    normalized_data = transforms.normalize(
        data, mean=data_mean, stddev=data_std
    )
    unnormalized_data = transforms.unnormalize(
        normalized_data, mean=data_mean, stddev=data_std
    )
    testing.assert_array_almost_equal(unnormalized_data, data)


if __name__ == '__main__':
  absltest.main()
