"""Implementation of UKAEA-TGLFNN as loaded from ONNX."""

from typing import Literal

import jax
import jax.numpy as jnp
import jaxonnxruntime
from jaxonnxruntime import backend as jaxort_backend
import onnx

from fusion_surrogates.ukaea_tglfnn import config as ukaea_tglfnn_config

jaxonnxruntime.config.update("jaxort_only_allow_initializers_as_static_args", False)


class ONNXTGLFNNModel:
    def __init__(
        self,
        efe_onnx_path: str,
        efi_onnx_path: str,
        pfi_onnx_path: str,
    ) -> "TGLFNNModel":

        self.models = {}

        efe_model = onnx.load_model(efe_onnx_path)
        self.models["efe_gb"] = jaxort_backend.Backend.prepare(efe_model)

        efi_model = onnx.load_model(efi_onnx_path)
        self.models["efi_gb"] = jaxort_backend.Backend.prepare(efi_model)

        pfi_model = onnx.load_model(pfi_onnx_path)
        self.models["pfi_gb"] = jaxort_backend.Backend.prepare(pfi_model)

        self._input_dtype = jnp.float32
        self._input_node_label = "input"

    def _predict_single_flux(
        self, flux: ukaea_tglfnn_config.OutputLabel, inputs: jax.Array
    ) -> jax.Array:
        output = self.models[flux].run(
            {self._input_node_label: inputs.astype(self._input_dtype)}
        )
        return jnp.stack([jnp.squeeze(output[0]), jnp.squeeze(output[1])], axis=-1)

    def predict(self, inputs: jax.Array):
        return jnp.stack(
            [
                self._predict_single_flux(f, inputs)
                for f in ukaea_tglfnn_config.OUTPUT_LABELS
            ],
            axis=-2,
        )
