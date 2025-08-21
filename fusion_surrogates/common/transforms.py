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

"""Common functions for transforming data."""

import jax
import jax.numpy as jnp


def normalize(
    data: jax.Array, *, mean: jax.Array, stddev: jax.Array
) -> jax.Array:
  """Normalizes data to have mean 0 and stddev 1."""
  return (data - mean) / jnp.where(stddev == 0, 1, stddev)


def unnormalize(
    data: jax.Array, *, mean: jax.Array, stddev: jax.Array
) -> jax.Array:
  """Unnormalizes data to the original distribution."""
  return data * jnp.where(stddev == 0, 1, stddev) + mean
