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

from typing import Any, Literal, Mapping, cast

from fusion_surrogates.common import networks
from fusion_surrogates.common import transforms
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
import tglfnn_ukaea as tglfnn_ukaea_lib

# Network version used for each machine when no version is given.
DEFAULT_VERSIONS = immutabledict.immutabledict({
    "multimachine": "3.2.1",
    "step": "1.0.1",
})


def _shared_value(model_dict: Mapping[str, Any], key: str) -> int:
  """The value of a per-output array that must be the same for every output."""
  values = np.unique(model_dict[key])
  if len(values) != 1:
    raise ValueError(
        f"{key} differs between outputs ({model_dict[key]}), which is not"
        " supported"
    )
  return int(values[0])


class TGLFNNukaeaModel:
  """UKAEA TGLF surrogate.

  This object is intended for use as static argument to jitted Jax
  functions. It therefore is immutable and hashes by value,
  not id.

  The parameters are exported from the original ONNX models, and the numerics
  follow them: the input and output normalisations are applied in float64
  (when JAX x64 is enabled) around the float32 network, and a small epsilon is
  added to each ensemble member's predicted variance.

  Attributes:
    machine: The machine the network was trained for.
    version: The version of the network, e.g. "3.2.1".
  """

  def __init__(
      self,
      machine: Literal["step", "multimachine"] = "multimachine",
      version: str | None = None,
  ):
    """Loads the network.

    Args:
      machine: The machine the network was trained for.
      version: The version of the network to load. If None, uses
        DEFAULT_VERSIONS[machine].
    """
    if machine not in DEFAULT_VERSIONS:
      raise ValueError(
          f"Unknown machine '{machine}' (must be one of"
          f" {', '.join(DEFAULT_VERSIONS)})"
      )
    self.machine = machine
    self.version = DEFAULT_VERSIONS[machine] if version is None else version
    model_dict = tglfnn_ukaea_lib.load(machine, version=self.version)
    self.input_labels = tuple(
        str(label) for label in model_dict["input_labels"]
    )
    self.output_labels = tuple(
        str(label) for label in model_dict["output_labels"]
    )
    # Each output has its own network, normalisations and variance epsilon,
    # stored with a leading axis ordered as output_labels.
    # Added to the variance predicted by each ensemble member, per output.
    self.variance_epsilon = np.asarray(
        model_dict["variance_epsilon"], dtype=np.float32
    )

    # The outputs and ensemble members are applied by vmapping a single
    # GaussianMLP over stacked parameters, so all outputs must share the same
    # architecture.
    self.n_ensemble = _shared_value(model_dict, "num_estimators")
    self._network = networks.GaussianMLP(
        num_hiddens=_shared_value(model_dict, "num_layers"),
        hidden_size=_shared_value(model_dict, "hidden_size"),
        dropout=0.0,  # Only used for training
        activation=str(model_dict["activation"]),
    )

    # Construct a PyTree of parameters
    # - Weights are stored per layer as (n_ensemble, out, in), i.e. transposed
    #   compared to the flax kernel
    # - "{output}/layer_{j}/{weight,bias}" -> "Dense_{j}/{kernel,bias}"
    # - Stacked for vmapping, with leading axes (n_outputs, n_ensemble)
    self._params = {
        f"Dense_{j}": {
            "kernel": jnp.stack([
                jnp.asarray(
                    model_dict[f"{label}/layer_{j}/weight"].transpose(0, 2, 1),
                    dtype=jnp.float32,
                )
                for label in self.output_labels
            ]),
            "bias": jnp.stack([
                jnp.asarray(
                    model_dict[f"{label}/layer_{j}/bias"], dtype=jnp.float32
                )
                for label in self.output_labels
            ]),
        }
        for j in range(self._network.num_hiddens)
    }

    # The input stats are (n_outputs, n_inputs) and the output stats
    # (n_outputs,). These are kept in float64 and cast to the normalisation
    # dtype in predict.
    self._input_means = np.asarray(model_dict["input_mean"], dtype=np.float64)
    self._input_stds = np.asarray(model_dict["input_std"], dtype=np.float64)
    self._output_means = np.asarray(model_dict["output_mean"], dtype=np.float64)
    self._output_stds = np.asarray(model_dict["output_std"], dtype=np.float64)

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

    # Normalise with each output's input stats, giving normalised inputs of
    # shape (n_outputs, ..., n_inputs).
    stats_axes = tuple(range(1, inputs.ndim))
    normalized_inputs = transforms.normalize(
        inputs.astype(stats_dtype)[None],
        mean=jnp.expand_dims(
            jnp.asarray(self._input_means, dtype=stats_dtype), stats_axes
        ),
        stddev=jnp.expand_dims(
            jnp.asarray(self._input_stds, dtype=stats_dtype), stats_axes
        ),
    ).astype(jnp.float32)

    def apply_member(params, member_inputs) -> jax.Array:
      # Module.apply is typed to also return the updated state, but only does
      # so when collections are mutable, which they are not here.
      return cast(
          jax.Array,
          self._network.apply(
              {"params": params}, member_inputs, deterministic=True
          ),
      )

    # Vmap over the output axis of the stacked parameters and normalised
    # inputs, and the ensemble axis of the parameters, giving predictions of
    # shape (n_outputs, n_ensemble, ..., 2).
    predictions = jax.vmap(
        lambda output_params, output_inputs: jax.vmap(
            apply_member, in_axes=(0, None)
        )(output_params, output_inputs)
    )(self._params, normalized_inputs)

    # Combine the ensemble members
    mean = jnp.mean(predictions[..., 0], axis=1)
    # Aleatoric uncertainty = mean of the predicted variances
    aleatoric = jnp.mean(predictions[..., 1], axis=1)
    # Epistemic uncertainty = variance of the predicted means
    epistemic = jnp.var(predictions[..., 0], axis=1)
    variance_epsilon = jnp.expand_dims(
        jnp.asarray(self.variance_epsilon), tuple(range(1, aleatoric.ndim))
    )
    normalized_predictions = jnp.stack(
        [mean, aleatoric + variance_epsilon + epistemic], axis=-1
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
    return hash((self.machine, self.version))

  def __eq__(self, other: Any) -> bool:
    return (
        isinstance(other, TGLFNNukaeaModel)
        and self.machine == other.machine
        and self.version == other.version
    )

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
