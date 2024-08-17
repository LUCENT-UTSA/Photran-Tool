import torch
import torch.nn as nn
from micro_ring_resonator import MicroRingResonator

class PhotonicCore(nn.Module):
    def __init__(self, input_size, output_size, resonator_radius, coupling_coeff, refractive_index):
        super(PhotonicCore, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.micro_ring_resonators = nn.ModuleList([
            MicroRingResonator(resonator_radius, coupling_coeff, refractive_index)
            for _ in range(input_size * output_size)
        ])

    def forward(self, input_matrix):
        output_matrix = torch.zeros(input_matrix.shape[0], self.output_size)
        for i in range(self.output_size):
            for j in range(self.input_size):
                voltage = input_matrix[:, j]
                transmission = self.micro_ring_resonators[i * self.input_size + j](voltage)
                output_matrix[:, i] += transmission
        return output_matrix
