import torch
from photonic_transformer import PhotonicTransformer

def calculate_power(transformer_model):
    total_power = 0.0
    for param in transformer_model.parameters():
        power = torch.sum(param.data**2)  # Placeholder: Replace with actual power calculation
        total_power += power.item()
    return total_power

# Example usage
model = PhotonicTransformer(input_size=64, hidden_size=128, output_size=10, resonator_radius=5e-6, coupling_coeff=0.5, refractive_index=1.5)
power = calculate_power(model)
print(f'Total Power Consumption: {power:.4f} W')
