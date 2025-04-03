"""Implementation of UKAEA-TGLFNN as loaded from Pytorch checkpoint."""

import jax
import jax.numpy as jnp
import optax
import torch

from fusion_surrogates import networks
from fusion_surrogates import transforms
from fusion_surrogates.ukaea_tglfnn import config as ukaea_tglfnn_config


def _convert_pytorch_state_dict(
    pytorch_state_dict: dict, config: ukaea_tglfnn_config.TGLFNNModelConfig
) -> optax.Params:
    params = {}
    for i in range(config.n_ensemble):
        model_dict = {}
        for j in range(config.num_hiddens):
            layer_dict = {
                "kernel": jnp.array(
                    pytorch_state_dict[f"models.{i}.model.{j*3}.weight"]
                ).T,
                "bias": jnp.array(pytorch_state_dict[f"models.{i}.model.{j*3}.bias"]).T,
            }
            model_dict[f"Dense_{j}"] = layer_dict
        params[f"GaussianMLP_{i}"] = model_dict
    return {"params": params}


class PytorchTGLFNNModel:
    def __init__(
        self,
        config_path: str,
        stats_path: str,
        efe_gb_checkpoint_path: str,
        efi_gb_checkpoint_path: str,
        pfi_gb_checkpoint_path: str,
        map_location: str = "cpu",
    ):
        self.config = ukaea_tglfnn_config.TGLFNNModelConfig.load(config_path)
        self.stats = ukaea_tglfnn_config.TGLFNNModelStats.load(stats_path)

        with open(efe_gb_checkpoint_path, "rb") as f:
            efe_gb_params = _convert_pytorch_state_dict(
                torch.load(f, map_location=map_location), self.config
            )
        with open(efi_gb_checkpoint_path, "rb") as f:
            efi_gb_params = _convert_pytorch_state_dict(
                torch.load(f, map_location=map_location), self.config
            )
        with open(pfi_gb_checkpoint_path, "rb") as f:
            pfi_gb_params = _convert_pytorch_state_dict(
                torch.load(f, map_location=map_location), self.config
            )

        self.params = {
            "efe_gb": efe_gb_params,
            "efi_gb": efi_gb_params,
            "pfi_gb": pfi_gb_params,
        }

        self.network = networks.GaussianMLPEnsemble(
            n_ensemble=self.config.n_ensemble,
            hidden_size=self.config.hidden_size,
            num_hiddens=self.config.num_hiddens,
            dropout=self.config.dropout,
            activation="relu",
        )

    def predict(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        """Compute the model prediction for the given inputs.

        Args:
            inputs: The input data to the model. Must be shape (..., 15).

        Returns:
            A jax.Array of shape (..., 3, 2), where output[..., i, 0]
            and output[..., i, 1] are the mean and variance for the ith flux output.
            Outputs are in the order of OUTPUT_LABELS, i.e. efe_gb, efi_gb, pfi_gb.
        """
        if self.config.normalize:
            inputs = transforms.normalize(
                inputs, mean=self.stats.input_mean, stddev=self.stats.input_std
            )

        output = jnp.stack(
            [
                self.network.apply(self.params[label], inputs, deterministic=True)
                for label in ukaea_tglfnn_config.OUTPUT_LABELS
            ],
            axis=-2,
        )

        if self.config.unnormalize:
            mean = output[..., 0]
            var = output[..., 1]

            unnormalized_mean = transforms.unnormalize(
                mean, mean=self.stats.output_mean, stddev=self.stats.output_std
            )

            output = jnp.stack([unnormalized_mean, var], axis=-1)

        return output
