# Copyright 2024 DeepMind Technologies Limited.
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

"""Base code for UKAEA's TGLFNN model."""

from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import optax

from fusion_surrogates.common import networks, transforms
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_config as config


class TGLFNNukaeaModel:
    """UKAEA TGLF surrogate."""

    def __init__(
        self,
        config: config.TGLFNNukaeaModelConfig,
        stats: config.TGLFNNukaeaModelStats,
        params: Mapping[config.OutputLabel, optax.Params] | None = None,
    ):
        self._config = config
        self._stats = stats
        self._params = params
        self._network = networks.GaussianMLPEnsemble(
            n_ensemble=config.n_ensemble,
            num_hiddens=config.num_hiddens,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            activation='relu',
        )

    def load_params(
        self, efe_gb_pt: str | Path, efi_gb_pt: str | Path, pfi_gb_pt: str | Path
    ) -> Mapping[config.OutputLabel, optax.Params]:
        self._params = {
            "efe_gb": config.params_from_pt_file(efe_gb_pt, self._config),
            "efi_gb": config.params_from_pt_file(efi_gb_pt, self._config),
            "pfi_gb": config.params_from_pt_file(pfi_gb_pt, self._config),
        }

    def predict(self, inputs: jax.Array) -> Mapping[config.OutputLabel, jax.Array]:
        """Predicts mean and variance of each flux.
        
        Internally normalizes the inputs based on the provided TGLFNNukaeaModelStats,
        applies the network, and denormalizes the outputs based on TGLFNNukaeaModelStats.
        """
        inputs = transforms.normalize(
            inputs,
            mean=self._stats.input_mean,
            stddev=self._stats.input_std,
        )

        predictions = {}

        for i, label in enumerate(config.OUTPUT_LABELS):
            prediction = self._network.apply(
                self._params[label], inputs, deterministic=True
            )

            mean_prediction = transforms.unnormalize(
                prediction[..., config.MEAN_OUTPUT_IDX],
                mean=self._stats.output_mean[i],
                stddev=self._stats.output_std[i],
            )
            variance_prediction = prediction[..., config.VAR_OUTPUT_IDX]
            prediction = jnp.stack([mean_prediction, variance_prediction], axis=-1)

            predictions[label] = prediction

        return predictions
