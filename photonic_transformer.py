import torch
import torch.nn as nn
from photonic_core import PhotonicCore

class PhotonicSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, resonator_radius, coupling_coeff, refractive_index):
        super(PhotonicSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query_proj = PhotonicCore(dim, dim, resonator_radius, coupling_coeff, refractive_index)
        self.key_proj = PhotonicCore(dim, dim, resonator_radius, coupling_coeff, refractive_index)
        self.value_proj = PhotonicCore(dim, dim, resonator_radius, coupling_coeff, refractive_index)
        self.out_proj = PhotonicCore(dim, dim, resonator_radius, coupling_coeff, refractive_index)

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        
        # Linear projections
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, dim)
        output = self.out_proj(attn_output)
        
        return output

class PhotonicTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, resonator_radius, coupling_coeff, refractive_index):
        super(PhotonicTransformerBlock, self).__init__()
        self.attention = PhotonicSelfAttention(dim, num_heads, resonator_radius, coupling_coeff, refractive_index)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            PhotonicCore(dim, mlp_dim, resonator_radius, coupling_coeff, refractive_index),
            nn.ReLU(),
            PhotonicCore(mlp_dim, dim, resonator_radius, coupling_coeff, refractive_index)
        )

    def forward(self, x):
        # Self-Attention and Add & Norm
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feedforward and Add & Norm
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        
        return x

class PhotonicTransformer(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, mlp_dim, output_size, resonator_radius, coupling_coeff, refractive_index):
        super(PhotonicTransformer, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Linear(input_size, input_size)

        self.layers = nn.ModuleList([
            PhotonicTransformerBlock(input_size, num_heads, mlp_dim, resonator_radius, coupling_coeff, refractive_index)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Initial embedding layer
        x = self.embedding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final output layer
        x = self.fc_out(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 64
    num_layers = 6
    num_heads = 8
    mlp_dim = 256
    output_size = 10
    resonator_radius = 5e-6
    coupling_coeff = 0.5
    refractive_index = 1.5

    # Dummy input (batch_size, sequence_length, input_size)
    x = torch.randn(32, 10, input_size)

    # Instantiate the model
    model = PhotonicTransformer(input_size, num_layers, num_heads, mlp_dim, output_size, resonator_radius, coupling_coeff, refractive_index)

    # Forward pass
    output = model(x)
    print(output.shape)
