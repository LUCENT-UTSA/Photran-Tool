import torch
import torch.nn as nn
from photonic_core import PhotonicCore

class PhotonicTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, resonator_radius, coupling_coeff, refractive_index):
        super(PhotonicTransformer, self).__init__()
        self.photonic_core = PhotonicCore(input_size, hidden_size, resonator_radius, coupling_coeff, refractive_index)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.photonic_core(x)
        x = self.fc_out(x)
        return x
