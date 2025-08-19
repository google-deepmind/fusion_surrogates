"""Script for generating the dummy torch checkpoint for testing TGLFNNukaea"""

import torch

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
