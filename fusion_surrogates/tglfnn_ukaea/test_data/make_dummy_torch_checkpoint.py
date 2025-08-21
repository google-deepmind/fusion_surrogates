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

"""Script for generating the dummy torch checkpoint for testing TGLFNNukaea."""

from absl import app
import torch


def main(_):
  input_dim = 13
  output_dim = 2
  hidden_dim = 5
  output_file = "dummy_torch_checkpoint.pt"

  state_dict = {
      # Ensemble member 1
      "models.0.model.0.weight": torch.zeros(hidden_dim, input_dim),
      "models.0.model.0.bias": torch.zeros(hidden_dim),
      "models.0.model.3.weight": torch.zeros(hidden_dim, hidden_dim),
      "models.0.model.3.bias": torch.zeros(hidden_dim),
      "models.0.model.6.weight": torch.zeros(output_dim, hidden_dim),
      "models.0.model.6.bias": torch.zeros(output_dim),
      # Ensemble member 2
      "models.1.model.0.weight": torch.zeros(hidden_dim, input_dim),
      "models.1.model.0.bias": torch.zeros(hidden_dim),
      "models.1.model.3.weight": torch.zeros(hidden_dim, hidden_dim),
      "models.1.model.3.bias": torch.zeros(hidden_dim),
      "models.1.model.6.weight": torch.zeros(output_dim, hidden_dim),
      "models.1.model.6.bias": torch.zeros(output_dim),
  }

  torch.save(state_dict, output_file)


if __name__ == "__main__":
  app.run(main)
