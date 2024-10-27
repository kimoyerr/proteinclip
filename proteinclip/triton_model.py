import torch
import triton

from proteinclip import triton_layers, triton_layer_norm_layer

# Define model using proteinclip architecture for Constrastive Embedding
class ContrastiveEmbeddingTriton(torch.nn.Module):
    def __init__(self, input_dim_1, input_dim_2, shared_dim):
        super(ContrastiveEmbeddingTriton, self).__init__()
        self.mlp_layer_1 = triton_layers.TritonLinearLayer(input_dim_1, input_dim_1, "gelu", bias=False)
        self.mlp_layer_1_shared = triton_layers.TritonLinearLayer(input_dim_1, shared_dim, None, bias=False)
        self.mlp_layer_1_norm = triton_layer_norm_layer.LayerNorm(input_dim_1, elementwise_affine=True)
        self.mlp_layer_2 = triton_layers.TritonLinearLayer(input_dim_2, input_dim_2, "gelu", bias=False)
        self.mlp_layer_2_shared = triton_layers.TritonLinearLayer(input_dim_2, shared_dim, None, bias=False)
        self.mlp_layer_2_norm = triton_layer_norm_layer.LayerNorm(input_dim_2, elementwise_affine=True)
    
    def forward(self, batch):
        # Mode 1
        mlp_layer_1_forward = self.mlp_layer_1(batch["x_1"])
        mlp_layer_1_forward_norm = self.mlp_layer_1_norm(mlp_layer_1_forward)
        x1_proj = self.mlp_layer_1_shared(mlp_layer_1_forward_norm)
        # Mode 2
        mlp_layer_2_forward = self.mlp_layer_2(batch["x_2"])
        mlp_layer_2_forward_norm = self.mlp_layer_2_norm(mlp_layer_2_forward)
        x2_proj = self.mlp_layer_2_shared(mlp_layer_2_forward_norm)
        return x1_proj, x2_proj


class ContrastiveEmbeddingTorch(torch.nn.Module):
    def __init__(self, input_dim_1, input_dim_2, shared_dim):
        super(ContrastiveEmbeddingTorch, self).__init__()
        self.mlp_layer_1 = torch.nn.Linear(input_dim_1, input_dim_1, bias=False)
        self.mlp_layer_1_shared = torch.nn.Linear(input_dim_1, shared_dim, bias=False)
        self.mlp_layer_1_norm = torch.nn.LayerNorm(input_dim_1, elementwise_affine=True)
        self.mlp_layer_2 = torch.nn.Linear(input_dim_2, input_dim_2, bias=False)
        self.mlp_layer_2_shared = torch.nn.Linear(input_dim_2, shared_dim, bias=False)
        self.mlp_layer_2_norm = torch.nn.LayerNorm(input_dim_2, elementwise_affine=True)
    
    def forward(self, batch):
        # Mode 1
        mlp_layer_1_forward = torch.nn.functional.gelu(self.mlp_layer_1(batch["x_1"]))
        mlp_layer_1_forward_norm = self.mlp_layer_1_norm(mlp_layer_1_forward)
        x1_proj = self.mlp_layer_1_shared(mlp_layer_1_forward_norm)
        # Mode 2
        mlp_layer_2_forward = torch.nn.functional.gelu(self.mlp_layer_2(batch["x_2"]))
        mlp_layer_2_forward_norm = self.mlp_layer_2_norm(mlp_layer_2_forward)
        x2_proj = self.mlp_layer_2_shared(mlp_layer_2_forward_norm)
        return x1_proj, x2_proj