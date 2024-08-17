import torch
import matplotlib.pyplot as plt
from micro_ring_resonator import MicroRingResonator

# Example data
input_voltages = torch.linspace(0, 10, 100)
resonator = MicroRingResonator(resonator_radius=5e-6, coupling_coeff=0.5, refractive_index=1.5)
outputs = resonator(input_voltages)

# Plot input vs. output
plt.figure()
plt.plot(input_voltages.numpy(), outputs.detach().numpy())
plt.title('Input Voltage vs. Micro Ring Resonator Output')
plt.xlabel('Input Voltage (V)')
plt.ylabel('Output Transmission')
plt.grid(True)
plt.show()

# Add additional plots as necessary
