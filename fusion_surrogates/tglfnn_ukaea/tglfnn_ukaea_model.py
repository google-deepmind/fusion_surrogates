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

from collections.abc import Mapping
from typing import Any, Literal

from fusion_surrogates.common import networks
from fusion_surrogates.common import transforms
import jax
import jax.numpy as jnp
import numpy as np
import tglfnn_ukaea as tglfnn_ukaea_lib


class TGLFNNukaeaModel:
  """UKAEA TGLF surrogate.

  This implementation is based on the original ONNX models.
  This object is intended for use as static argument to jitted JAX
  functions. It therefore is immutable and hashes by value,
  not id.
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
        `tglfnn_ukaea.DEFAULT_VERSIONS[machine]`.
    """
    self._machine: Literal["step", "multimachine"] = machine
    model_dict = tglfnn_ukaea_lib.load(machine, version=version)
    self._version: str = model_dict["version"].item()
    self._input_labels: tuple[str, ...] = tuple(
        model_dict["input_labels"].tolist()
    )
    self._output_labels: tuple[str, ...] = tuple(
        model_dict["output_labels"].tolist()
    )
    self._variance_epsilon: jax.Array = jnp.asarray(
        model_dict["variance_epsilon"], dtype=jnp.float32
    )

    # The outputs and ensemble members are applied by vmapping a single
    # GaussianMLP over stacked parameters, so all outputs must share the same
    # architecture.
    for key in ("num_estimators", "num_layers", "hidden_size"):
      if len(np.unique(model_dict[key])) != 1:
        raise ValueError(
            f"{key} differs between outputs ({model_dict[key]}), which is not"
            " supported"
        )
    self._n_ensemble: int = model_dict["num_estimators"][0].item()
    self._network = networks.GaussianMLP(
        num_hiddens=model_dict["num_layers"][0].item(),
        hidden_size=model_dict["hidden_size"][0].item(),
        dropout=0.0,  # Only used for training
        activation=model_dict["activation"].item(),
    )

    # Construct a PyTree of parameters
    # - Weights are stored per layer as (n_ensemble, out, in), i.e. transposed
    #   compared to the flax kernel
    # - "{output}/layer_{j}/{weight,bias}" -> "Dense_{j}/{kernel,bias}"
    # - Stacked for vmapping, with leading axes (n_outputs, n_ensemble)
    # - Stack on CPU in numpy, then convert to JAX arrays in float32.
    self._params = {
        f"Dense_{j}": {
            "kernel": jnp.asarray(
                np.stack([
                    model_dict[f"{label}/layer_{j}/weight"]
                    for label in self.output_labels
                ]).swapaxes(-1, -2),
                dtype=jnp.float32,
            ),
            "bias": jnp.asarray(
                np.stack([
                    model_dict[f"{label}/layer_{j}/bias"]
                    for label in self.output_labels
                ]),
                dtype=jnp.float32,
            ),
        }
        for j in range(self._network.num_hiddens)
    }

    # The input stats are (n_outputs, n_inputs) and the output stats
    # (n_outputs,), stored in float64 to match the ONNX models.
    self._input_means = jnp.asarray(model_dict["input_mean"], dtype=jnp.float64)
    self._input_stds = jnp.asarray(model_dict["input_std"], dtype=jnp.float64)
    self._output_means = jnp.asarray(
        model_dict["output_mean"], dtype=jnp.float64
    )
    self._output_stds = jnp.asarray(model_dict["output_std"], dtype=jnp.float64)

    # __init__ method is done, activate immutability
    self._frozen = True

  @property
  def machine(self) -> Literal["step", "multimachine"]:
    """The machine the network was trained for."""
    return self._machine

  @property
  def version(self) -> str:
    """TGLFNN-UKAEA model version number."""
    return self._version

  @property
  def input_labels(self) -> tuple[str, ...]:
    """Ordered tuple of input feature names."""
    return self._input_labels

  @property
  def output_labels(self) -> tuple[str, ...]:
    """Ordered tuple of output flux names."""
    return self._output_labels

  @property
  def variance_epsilon(self) -> jax.Array:
    """Per-output epsilon added to each ensemble member's predicted variance."""
    return self._variance_epsilon

  @property
  def n_ensemble(self) -> int:
    """Number of ensemble members per output."""
    return self._n_ensemble

  def predict(self, inputs: jax.Array) -> Mapping[str, jax.Array]:
    """Predicts the unnormalised mean and variance of each output flux.

    Args:
      inputs: Unnormalised input features of shape `(..., n_inputs)`, ordered
        as `self.input_labels`, where `n_inputs == len(self.input_labels)`.

    Returns:
      A mapping from each flux label in `self.output_labels` to a `float64`
      array of shape `(..., 2)` containing `[mean, variance]` along the
      trailing axis.
    """
    # Vmap over the output axis of the stacked parameters and per-output
    # statistics, giving predictions of shape (n_outputs, ..., 2).
    predictions = jax.vmap(
        self._predict_single_output,
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )(
        inputs,
        self._params,
        self._variance_epsilon,
        self._input_means,
        self._input_stds,
        self._output_means,
        self._output_stds,
    )

    return {label: predictions[i] for i, label in enumerate(self.output_labels)}

  def _predict_single_output(
      self,
      inputs: jax.Array,
      params: Mapping[str, Mapping[str, jax.Array]],
      variance_epsilon: jax.Array,
      input_mean: jax.Array,
      input_std: jax.Array,
      output_mean: jax.Array,
      output_std: jax.Array,
  ) -> jax.Array:
    """Predicts unnormalised [mean, variance] for a single output flux."""
    # Like the ONNX models, input and output normalisations run in float64
    # around the float32 network.
    normalized_inputs = transforms.normalize(
        inputs, mean=input_mean, stddev=input_std
    ).astype(jnp.float32)

    # Vmap over the ensemble axis of this output's parameters, giving
    # predictions of shape (n_ensemble, ..., 2).
    member_predictions = jax.vmap(
        lambda p: self._network.apply(
            {"params": p}, normalized_inputs, deterministic=True
        )
    )(params)
    assert isinstance(member_predictions, jax.Array)

    # Combine the ensemble members
    member_means = member_predictions[..., 0]
    mean = jnp.mean(member_means, axis=0)
    # Aleatoric uncertainty = mean of the predicted variances (plus epsilon)
    aleatoric = jnp.mean(member_predictions[..., 1], axis=0) + variance_epsilon
    # Epistemic uncertainty = variance of the predicted means
    epistemic = jnp.mean(jnp.square(member_means - mean), axis=0)

    mean = transforms.unnormalize(mean, mean=output_mean, stddev=output_std)
    variance = transforms.unnormalize(
        aleatoric + epistemic,
        mean=jnp.zeros_like(output_mean),
        stddev=output_std**2,
    )
    return jnp.stack([mean, variance], axis=-1)

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
