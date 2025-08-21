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

"""Registry of paths for QLKNN models."""
import pathlib
import immutabledict

DEFAULT_MODEL_NAME = 'qlknn_7_11_v1'

MODELS = immutabledict.immutabledict({
    'qlknn_7_11_v1': f'{pathlib.Path(__file__).parent}/qlknn_7_11.qlknn',
})

ONNX_MODELS = immutabledict.immutabledict({
    'qlknn_7_11_v1': f'{pathlib.Path(__file__).parent}/qlknn_7_11.onnx',
})
