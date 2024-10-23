# Inspired from https://github.com/BobMcDear/attorch/blob/main/attorch/linear_layer.py
import typing
import torch
from torch import nn
from triton import cdiv

from proteinclip import triton_kernels
from proteinclip.triton_utils import get_output_dtype


class TritonLinearAutograd(torch.autograd.Function):
    """
    Custom autograd function for linear layer using Triton kernels
    """
    @staticmethod
    def forward(
        ctx: typing.Any,
        inputs: torch.Tensor,
        weights: typing.Optional[torch.Tensor] = None,
        bias: typing.Optional[torch.Tensor] = None,
        act_func: typing.Optional[str] = None,  
    ) -> torch.Tensor:
        """
        Linearly transforms the input using weights, optionally adding bias and fusing an activation function.
        """

        # Assert that the weights are 2D
        assert weights.ndim == 2, f'Weights must be 2D, received shape {weights.shape}'
        # Assert that the bias is 1D
        assert bias is None or bias.ndim == 1, f'Bias must be 1D, received shape {bias.shape}'
        # Assert that the input and weights are compatible
        assert inputs.shape[-1] == weights.shape[0], f'Incompatible input ({inputs.shape}) and weights ({weights.shape}) shape'
        # Assert that the weights and bias are compatible
        assert bias is None or weights.shape[1] == bias.shape[0], f'Incompatible weights ({weights.shape}) and bias ({bias.shape}) shape'
        
        # IF actiavtion function is None, set it to 'gelu'
        param = None
        if act_func is None:
            act_func = 'gelu'

        # If weights are None, throw an error
        if weights is None:
            raise ValueError("Weights must be provided")


        flattened_inputs = inputs.flatten(0, -2)
        batch_dim, in_feat_dim = flattened_inputs.shape
        _, out_feat_dim = weights.shape

        requires_grad = (inputs.requires_grad or weights.requires_grad or (bias is not None and bias.requires_grad))
        # Only save pre-activation if we need to backprop through the activation function
        save_pre_act = requires_grad and (act_func is not None)

        # Create an empty torch tensor for the output
        outputs_dtype = get_output_dtype(inputs.dtype, autocast='fp16')
        outputs = torch.empty((batch_dim, out_feat_dim), dtype=outputs_dtype, device=inputs.device)
        pre_act = torch.empty_like(outputs) if save_pre_act else outputs

        # Launches a 1D grid, where each program outputs blocks of
        # BLOCK_SIZE_BATCH rows and BLOCK_SIZE_OUT_FEAT columns.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']) *
                             cdiv(out_feat_dim, META['BLOCK_SIZE_OUT_FEAT']),)
        triton_kernels.triton_linear_forward_kernel[grid](
            flattened_inputs, 
            weights, 
            inputs if bias is None else bias,
            pre_act,
            outputs,
            batch_dim, 
            in_feat_dim, 
            out_feat_dim,
            *flattened_inputs.stride(),
            *weights.stride(),
            *pre_act.stride(),
            *outputs.stride(),
            param,
            add_bias=bias is not None,
            act_func=act_func,
            save_pre_act=save_pre_act,
            fp16=outputs_dtype is torch.float16
        )
        
        # Save the context
        ctx.param = param
        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.outputs_dtype = outputs_dtype
        if requires_grad:
            ctx.save_for_backward(inputs, pre_act if save_pre_act else None, weights)

        return outputs.view(*inputs.shape[:-1], out_feat_dim)


    @staticmethod
    def backward(
        ctx: typing.Any,
        output_grad: torch.Tensor,
    ):
        inputs, pre_act, weights = ctx.saved_tensors

        flattened_inputs = inputs.flatten(0, -2)
        batch_dim, in_feat_dim = flattened_inputs.shape
        _, out_feat_dim = weights.shape

        if ctx.act_func is None:
            pre_act_grad = output_grad
        # TODO: Implement the activation function
        else:
            pre_act_grad = output_grad

        with torch.autocast("cuda", ctx.output_dtype):
            inputs_grad = pre_act_grad @ weights.T if inputs.requires_grad else None
            weights_grad = flattened_inputs.T @ pre_act_grad if weights.requires_grad else None

        return inputs_grad.view_as(inputs), weights_grad, None, None
        

class TritonLinearLayer(nn.Linear):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        act_func: str,
        bias: bool = True,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(input_dim, output_dim, bias, device, dtype)
        self.weight = nn.Parameter(self.weight.T.contiguous())
        self.act_func = act_func
        
        # KaiMing Initialization
        if self.act_func == 'gelu':
            nonlinearity = "relu"
        else:
            nonlinearity = self.act_func

        torch.nn.init.kaiming_normal_(self.weight, nonlinearity=nonlinearity, mode='fan_in')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return TritonLinearAutograd.apply(input, self.weight, self.bias, self.act_func)



