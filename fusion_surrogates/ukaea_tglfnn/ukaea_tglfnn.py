import dataclasses
import json
from typing import Final

import jax
import jax.numpy as jnp
import optax
import yaml

from fusion_surrogates.networks import GaussianMLPEnsemble
from fusion_surrogates.utils import normalize, unnormalize

INPUT_LABELS: Final[list[str]] = [
    "RLNS_1",
    "RLTS_1",
    "RLTS_2",
    "TAUS_2",
    "RMIN_LOC",
    "DRMAJDX_LOC",
    "Q_LOC",
    "SHAT",
    "XNUE",
    "KAPPA_LOC",
    "S_KAPPA_LOC",
    "DELTA_LOC",
    "S_DELTA_LOC",
    "BETAE",
    "ZEFF",
]
OUTPUT_LABELS: Final[list[str]] = ["efe_gb", "efi_gb", "pfi_gb"]


@dataclasses.dataclass
class TGLFNNModelConfig:
    n_ensemble: int
    num_hiddens: int
    dropout: float
    normalize: bool = True
    unnormalize: bool = True
    hidden_size: int = 512

    @classmethod
    def load(cls, config_path: str) -> "TGLFNNModelConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(
            n_ensemble=config["num_estimators"],
            num_hiddens=config["model_size"],
            dropout=config["dropout"],
        )


@dataclasses.dataclass
class TGLFNNModelStats:
    input_mean: jax.Array
    input_std: jax.Array
    output_mean: jax.Array
    output_std: jax.Array

    @classmethod
    def load(cls, stats_path: str) -> "TGLFNNModelStats":
        with open(stats_path, "r") as f:
            stats = json.load(f)

        return cls(
            input_mean=jnp.array([stats[label]["mean"] for label in INPUT_LABELS]),
            input_std=jnp.array([stats[label]["std"] for label in INPUT_LABELS]),
            output_mean=jnp.array([stats[label]["mean"] for label in OUTPUT_LABELS]),
            output_std=jnp.array([stats[label]["std"] for label in OUTPUT_LABELS]),
        )


class TGLFNNModel:

    def __init__(
        self,
        config: TGLFNNModelConfig,
        stats: TGLFNNModelStats,
        params: optax.Params | None,
    ):
        self.config = config
        self.stats = stats
        self.params = params
        self.network = GaussianMLPEnsemble(
            n_ensemble=config.n_ensemble,
            hidden_size=config.hidden_size,
            num_hiddens=config.num_hiddens,
            dropout=config.dropout,
            activation="relu",
        )

    @classmethod
    def load_from_pytorch(
        cls,
        config_path: str,
        stats_path: str,
        efe_gb_checkpoint_path: str,
        efi_gb_checkpoint_path: str,
        pfi_gb_checkpoint_path: str,
        *args,
        **kwargs,
    ) -> "TGLFNNModel":
        import torch

        def _convert_pytorch_state_dict(
            pytorch_state_dict: dict, config: TGLFNNModelConfig
        ) -> optax.Params:
            params = {}
            for i in range(config.n_ensemble):
                model_dict = {}
                for j in range(config.num_hiddens):
                    layer_dict = {
                        "kernel": jnp.array(
                            pytorch_state_dict[f"models.{i}.model.{j*3}.weight"]
                        ).T,
                        "bias": jnp.array(
                            pytorch_state_dict[f"models.{i}.model.{j*3}.bias"]
                        ).T,
                    }
                    model_dict[f"Dense_{j}"] = layer_dict
                params[f"GaussianMLP_{i}"] = model_dict
            return {"params": params}

        config = TGLFNNModelConfig.load(config_path)
        stats = TGLFNNModelStats.load(stats_path)

        with open(efe_gb_checkpoint_path, "rb") as f:
            efe_gb_params = _convert_pytorch_state_dict(
                torch.load(f, *args, **kwargs), config
            )
        with open(efi_gb_checkpoint_path, "rb") as f:
            efi_gb_params = _convert_pytorch_state_dict(
                torch.load(f, *args, **kwargs), config
            )
        with open(pfi_gb_checkpoint_path, "rb") as f:
            pfi_gb_params = _convert_pytorch_state_dict(
                torch.load(f, *args, **kwargs), config
            )

        params = {
            "efe_gb": efe_gb_params,
            "efi_gb": efi_gb_params,
            "pfi_gb": pfi_gb_params,
        }

        return cls(config, stats, params)

    def predict(
        self,
        inputs: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute the model prediction for the given inputs.

        Args:
            inputs: The input data to the model. Must be shape (..., 15).

        Returns:
            A jax.Array of shape (..., 3, 2), where output[..., i, 0]
            and output[..., i, 1] are the mean and variance for the ith flux output.
            Outputs are in the order of OUTPUT_LABELS, i.e. efe_gb, efi_gb, pfi_gb.
        """
        if self.config.normalize:
            inputs = normalize(
                inputs, mean=self.stats.input_mean, stddev=self.stats.input_std
            )

        output = jnp.stack(
            [
                self.network.apply(self.params[label], inputs, deterministic=True)
                for label in OUTPUT_LABELS
            ],
            axis=-2,
        )

        if self.config.unnormalize:
            mean = output[..., 0]
            var = output[..., 1]

            unnormalized_mean = unnormalize(
                mean, mean=self.stats.output_mean, stddev=self.stats.output_std
            )

            output = jnp.stack([unnormalized_mean, var], axis=-1)

        return output


class ONNXTGLFNNModel:
    def __init__(
        self,
        efe_onnx_path: str,
        efi_onnx_path: str,
        pfi_onnx_path: str,
    ) -> "TGLFNNModel":
        import onnx
        from jaxonnxruntime import config
        from jaxonnxruntime.backend import Backend as ONNXJaxBackend

        config.update("jaxort_only_allow_initializers_as_static_args", False)

        self.models = {}
        efe_model = onnx.load_model(efe_onnx_path)
        self.models["efe_gb"] = ONNXJaxBackend.prepare(efe_model)

        efi_model = onnx.load_model(efi_onnx_path)
        self.models["efi_gb"] = ONNXJaxBackend.prepare(efi_model)

        pfi_model = onnx.load_model(pfi_onnx_path)
        self.models["pfi_gb"] = ONNXJaxBackend.prepare(pfi_model)

        self._input_dtype = jnp.float32
        self._input_node_label = "input"

    def _predict_single_flux(self, flux: str, inputs: jax.Array) -> jax.Array:
        output = self.models[flux].run(
            {self._input_node_label: inputs.astype(self._input_dtype)}
        )
        return jnp.stack([jnp.squeeze(output[0]), jnp.squeeze(output[1])], axis=-1)

    def predict(self, inputs: jax.Array):
        return jnp.stack(
            [self._predict_single_flux(f, inputs) for f in OUTPUT_LABELS], axis=-2
        )
