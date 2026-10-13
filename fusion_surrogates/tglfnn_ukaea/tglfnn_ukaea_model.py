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

from typing import Any, Literal, Mapping

from fusion_surrogates.common import networks
from fusion_surrogates.common import transforms
import jax
import jax.numpy as jnp
import numpy as np
import tglfnn_ukaea as tglfnn_ukaea_lib

# Added to the variance predicted by each ensemble member, as in the original
# ONNX/TorchScript models.
_VARIANCE_EPSILON = 1e-6


class TGLFNNukaeaModel:
  """UKAEA TGLF surrogate.

  This object is intended for use as static argument to jitted Jax
  functions. It therefore is immutable and hashes by value,
  not id.

  The numerics follow the original ONNX/TorchScript models: the input and
  output normalisations are applied in float64 (when JAX x64 is enabled) around
  the float32 network, and a small epsilon is added to each ensemble member's
  predicted variance.
  """

  def __init__(
      self,
      machine: Literal["step", "multimachine"] = "multimachine",
  ):
    self.machine = machine
    model_dict = tglfnn_ukaea_lib.load(machine)

    self.input_labels = model_dict["input_labels"]
    self.output_labels = tuple(model_dict["params"].keys())

    # The ensemble is applied by vmapping a single GaussianMLP over stacked
    # member parameters.
    self.n_ensemble = model_dict["config"].get("num_estimators", 5)
    self._network = networks.GaussianMLP(
        num_hiddens=model_dict["config"].get("model_size", 6),
        hidden_size=model_dict["config"].get("hidden_size", 512),
        dropout=model_dict["config"].get("dropout", 0.0),
        activation=model_dict["config"].get("activation", "relu"),
    )

    # Construct a PyTree of parameters
    # - Transposed weights compared to the original model
    # - The following label changes:
    #   "FullyConnectedLayer_{i}" -> "Dense_{i}"
    #   "weight" -> "kernel"
    # - Stacked for vmapping, with leading axes (n_outputs, n_ensemble)
    params = {}
    for output_label in self.output_labels:
      ensemble = []
      for i in range(self.n_ensemble):
        network_params = {}
        for j in range(self._network.num_hiddens):
          layer_params = {}
          original_layer_params = model_dict["params"][output_label][
              f"MLP_{i}"
          ][f"FullyConnectedLayer_{j}"]
          layer_params["bias"] = jnp.asarray(
              original_layer_params["bias"].T, dtype=jnp.float32
          )
          layer_params["kernel"] = jnp.asarray(
              original_layer_params["weight"].T, dtype=jnp.float32
          )
          network_params[f"Dense_{j}"] = layer_params
        ensemble.append(network_params)
      # Stack the parameters of all ensemble members for this output
      params[output_label] = jax.tree.map(
          lambda *args: jnp.stack(args), *ensemble
      )
    # Stack the outputs
    self._params = jax.tree.map(
        lambda *args: jnp.stack(args),
        *[params[label] for label in self.output_labels],
    )

    # Vectorize the stats in label order. These are kept in float64 and cast
    # to the normalisation dtype in predict.
    stats = model_dict["stats"]
    self._input_means = np.array(
        [stats[label]["mean"] for label in self.input_labels], dtype=np.float64
    )
    self._input_stds = np.array(
        [stats[label]["std"] for label in self.input_labels], dtype=np.float64
    )
    self._output_means = np.array(
        [stats[label]["mean"] for label in self.output_labels], dtype=np.float64
    )
    self._output_stds = np.array(
        [stats[label]["std"] for label in self.output_labels], dtype=np.float64
    )

    # __init__ method is done, activate immutability
    self._frozen = True

  def predict(self, inputs: jax.Array) -> Mapping[str, jax.Array]:
    """Predicts mean and variance of each flux."""
    input_dtype = inputs.dtype
    # Normalise in float64 if available (otherwise float32), and run the
    # network in float32.
    stats_dtype = jnp.promote_types(
        input_dtype, jax.dtypes.canonicalize_dtype(jnp.float64)
    )

    normalized_inputs = transforms.normalize(
        inputs.astype(stats_dtype),
        mean=jnp.asarray(self._input_means, dtype=stats_dtype),
        stddev=jnp.asarray(self._input_stds, dtype=stats_dtype),
    ).astype(jnp.float32)

    # Vmap over the output and ensemble axes of the stacked parameters,
    # giving predictions of shape (n_outputs, n_ensemble, ..., 2).
    predictions = jax.vmap(
        jax.vmap(
            lambda params: self._network.apply(
                {"params": params},
                normalized_inputs,
                deterministic=True,
            ),
        )
    )(self._params)

    # Combine the ensemble members
    mean = jnp.mean(predictions[..., 0], axis=1)  # pyrefly: ignore[bad-index]
    # Aleatoric uncertainty = mean of the predicted variances
    aleatoric = jnp.mean(
        predictions[..., 1], axis=1
    )  # pyrefly: ignore[bad-index]
    # Epistemic uncertainty = variance of the predicted means
    epistemic = jnp.var(
        predictions[..., 0], axis=1
    )  # pyrefly: ignore[bad-index]
    normalized_predictions = jnp.stack(
        [mean, aleatoric + _VARIANCE_EPSILON + epistemic], axis=-1
    ).astype(stats_dtype)

    # Variances have no mean offset and scale with the squared stddev.
    output_means = jnp.asarray(self._output_means, dtype=stats_dtype)
    output_stds = jnp.asarray(self._output_stds, dtype=stats_dtype)
    broadcast_means = jnp.expand_dims(
        jnp.stack([output_means, jnp.zeros_like(output_means)], axis=-1),
        axis=tuple(range(1, normalized_predictions.ndim - 1)),
    )
    broadcast_stds = jnp.expand_dims(
        jnp.stack([output_stds, output_stds**2], axis=-1),
        axis=tuple(range(1, normalized_predictions.ndim - 1)),
    )

    # Unnormalize and explicitly cast back to original dtype.
    predictions = transforms.unnormalize(
        normalized_predictions, mean=broadcast_means, stddev=broadcast_stds
    ).astype(input_dtype)

    return {label: predictions[i] for i, label in enumerate(self.output_labels)}

  def __hash__(self) -> int:
    return hash(self.machine)

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, TGLFNNukaeaModel) and self.machine == other.machine

  def __setattr__(self, attr, value):
    # pylint: disable=g-doc-args
    # pylint: disable=g-doc-return-or-yield
    """Override __setattr__ to make the class (sort of) immutable.

    Note that you can still do obj.field.subfield = x, so it is not true
    immutability, but this to helps to avoid some careless errors.
    """
    if getattr(self, "_frozen", False):
      raise AttributeError("TGLFNNukaeaModel is immutable.")
    return super().__setattr__(attr, value)
