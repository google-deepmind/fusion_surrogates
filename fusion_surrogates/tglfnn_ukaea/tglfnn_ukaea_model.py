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

# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX implementation of UKAEA's TGLFNN model."""

from typing import Literal, Mapping

from fusion_surrogates.common import networks
from fusion_surrogates.common import transforms
import jax
import jax.numpy as jnp
import tglfnn_ukaea as tglfnn_ukaea_lib


class TGLFNNukaeaModel:
  """UKAEA TGLF surrogate."""

  def __init__(
      self,
      machine: Literal["step", "multimachine"] = "multimachine",
  ):
    model_dict = tglfnn_ukaea_lib.load(machine)

    self.input_labels = model_dict["input_labels"]
    self.output_labels = tuple(model_dict["params"].keys())

    self._network = networks.GaussianMLPEnsemble(
        n_ensemble=model_dict["config"].get("num_estimators", 5),
        num_hiddens=model_dict["config"].get("model_size", 6),
        hidden_size=model_dict["config"].get("hidden_size", 512),
        dropout=model_dict["config"].get("dropout", 0.0),
        activation=model_dict["config"].get("activation", "relu"),
    )

    # Construct a PyTree of parameters
    # - Transposed weights compared to the original model
    # - The following label changes:
    #   "MLP_{i}" -> "GaussianMLP_{i}"
    #   "FullyConnectedLayer_{i}" -> "Dense_{i}"
    #   "weight" -> "kernel"
    # - Stacked for vmapping
    params = {}
    for output_label in self.output_labels:
      ensemble = {}
      for i in range(self._network.n_ensemble):
        network_params = {}
        for j in range(self._network.num_hiddens):
          layer_params = {}
          original_layer_params = model_dict["params"][output_label][
              f"MLP_{i}"
          ][f"FullyConnectedLayer_{j}"]
          layer_params["bias"] = jnp.array(original_layer_params["bias"].T)
          layer_params["kernel"] = jnp.array(original_layer_params["weight"].T)
          network_params[f"Dense_{j}"] = layer_params
        ensemble[f"GaussianMLP_{i}"] = network_params
      params[output_label] = ensemble
    # Stack the parameters
    self._params = jax.tree.map(
        lambda *args: jnp.stack(args),
        *[params[label] for label in self.output_labels],
    )

    # Vectorize the stats
    self._input_means = jnp.array([
        v["mean"]
        for k, v in model_dict["stats"].items()
        if k in self.input_labels
    ])
    self._input_stds = jnp.array([
        v["std"]
        for k, v in model_dict["stats"].items()
        if k in self.input_labels
    ])
    self._output_means = jnp.array([
        v["mean"]
        for k, v in model_dict["stats"].items()
        if k in self.output_labels
    ])
    self._output_stds = jnp.array([
        v["std"]
        for k, v in model_dict["stats"].items()
        if k in self.output_labels
    ])

  def predict(self, inputs: jax.Array) -> Mapping[str, jax.Array]:
    """Predicts mean and variance of each flux."""
    normalized_inputs = transforms.normalize(
        inputs,
        mean=self._input_means,
        stddev=self._input_stds,
    )

    normalized_predictions = jax.vmap(
        lambda params: self._network.apply(
            {"params": params},
            normalized_inputs,
            deterministic=True,
        ),
    )(self._params)

    broadcast_means = jnp.expand_dims(
        self._output_means,
        axis=tuple(range(1, normalized_predictions.ndim)),
    )
    broadcast_stds = jnp.expand_dims(
        self._output_stds,
        axis=tuple(range(1, normalized_predictions.ndim)),
    )
    predictions = transforms.unnormalize(
        normalized_predictions, mean=broadcast_means, stddev=broadcast_stds
    )

    return {label: predictions[i] for i, label in enumerate(self.output_labels)}
