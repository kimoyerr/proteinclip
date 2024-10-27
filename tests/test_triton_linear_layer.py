import pytest
import torch

from proteinclip import triton_layers


test_data = [
    (16, 32, 256, 2048),
    (16, 64, 512, 2048),
    (16, 64, 512, 2048),
    (16, 256, 2048, 4096),

]
@pytest.mark.parametrize("batch_dim, in_dim, mlp_dim, shared_dim", test_data)
def test_triton_linear_layer_correctness(batch_dim, in_dim, mlp_dim, shared_dim):

    # Gererate random data
    tmp_batch = torch.rand(batch_dim, in_dim).to(torch.device('cuda'))

    # Triton custom linear layer
    act_func = "gelu"
    mlp_triton_layer = triton_layers.TritonLinearLayer(in_dim, mlp_dim, "gelu", bias=False)
    mlp_triton_layer_forward = mlp_triton_layer(tmp_batch)
    
    # Layer comparison with Torch nn
    mlp_torch_layer = torch.nn.Linear(in_dim, mlp_dim).to(torch.device('cuda'))
    mlp_torch_layer.weight = torch.nn.Parameter(mlp_triton_layer.weight.t())   # Transpose the weight
    mlp_torch_layer.bias = None
    mlp_torch_layer_forward = mlp_torch_layer(tmp_batch)
    # Gelu
    mlp_torch_layer_gelu_forward = torch.nn.functional.gelu(mlp_torch_layer_forward)
    
    # Check the forward pass
    assert torch.allclose(mlp_triton_layer_forward, mlp_torch_layer_gelu_forward, rtol=1e-2, atol=1e-2)
    # False assertion for testing gelu output has to be different
    assert not torch.allclose(mlp_triton_layer_forward, mlp_torch_layer_forward, rtol=1e-2, atol=1e-2)


    # Second layer
    # Triton
    mlp_triton_shared_layer = triton_layers.TritonLinearLayer(mlp_dim, shared_dim, None, bias=False)  
    mlp_triton_shared_layer_forward = mlp_triton_shared_layer(mlp_triton_layer_forward)
    # Torch
    mlp_torch_shared_layer = torch.nn.Linear(mlp_dim, shared_dim).to(torch.device('cuda'))
    mlp_torch_shared_layer.weight = torch.nn.Parameter(mlp_triton_shared_layer.weight.t())
    mlp_torch_shared_layer.bias = None
    mlp_torch_shared_layer_forward = mlp_torch_shared_layer(mlp_torch_layer_gelu_forward)
    
    # Check
    assert torch.allclose(mlp_triton_shared_layer_forward, mlp_torch_shared_layer_forward, rtol=1e-2, atol=1e-2)

    # Backward
    triton_loss = torch.sum(mlp_triton_shared_layer_forward)
    triton_loss.backward()
    torch_loss = torch.sum(mlp_torch_shared_layer_forward)
    torch_loss.backward()

    # # Check the gradients
    assert torch.allclose(mlp_triton_layer.weight.grad, mlp_torch_layer.weight.grad.t(), rtol=1e-2, atol=1e-2)
    assert torch.allclose(mlp_triton_shared_layer.weight.grad, mlp_torch_shared_layer.weight.grad.t(), rtol=1e-2, atol=1e-2)


