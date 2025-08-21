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

"""Utils for qlknn_model tests."""

from fusion_surrogates.common import networks
from fusion_surrogates.qlknn import qlknn_model
import jax


def _get_test_stats_data(
    num_inputs: int, num_targets: int
) -> qlknn_model.QLKNNStatsData:
  key = jax.random.PRNGKey(1234)
  return qlknn_model.QLKNNStatsData(
      input_mean=jax.random.uniform(key, (num_inputs,)),
      input_stddev=jax.random.uniform(key, (num_inputs,)),
      target_mean=jax.random.uniform(key, (num_targets,)),
      target_stddev=jax.random.uniform(key, (num_targets,)),
      input_min=jax.random.uniform(key, (num_inputs,)),
      input_max=jax.random.uniform(key, (num_inputs,)),
  )


def get_test_flux_map() -> dict[str, dict[str, str | None]]:
  return {
      'flux0': {'target': 'out0', 'denominator': None},
      'flux1': {'target': 'out1', 'denominator': 'out0'},
      'flux2': {'target': 'out2', 'denominator': 'out0'},
  }


def get_test_model_config() -> qlknn_model.QLKNNModelConfig:
  num_inputs = 4
  num_targets = 3
  return qlknn_model.QLKNNModelConfig(
      normalize_inputs=True,
      normalize_targets=True,
      input_names=['Ate', 'Ati', 'in2', 'in3'],
      target_names=['out0', 'out1', 'out2'],
      stats_data=_get_test_stats_data(num_inputs, num_targets),
      flux_map=get_test_flux_map(),
      network_type=networks.NetworkType.MLP,
      network_config=qlknn_model.MLPConfig(
          activation='tanh',
          num_hiddens=3,
          hidden_size=10,
      ),
  )


def init_model(
    config: qlknn_model.QLKNNModelConfig, batch_dims: tuple[int, int]
):
  init_rng = jax.random.split(jax.random.key(1), batch_dims[0])
  model = qlknn_model.QLKNNModel(config=config)
  model.init_params(batch_dims=batch_dims, init_rng=init_rng)
  return model
