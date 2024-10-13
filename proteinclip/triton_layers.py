import os

import typing
import torch
from torch import nn
import triton
from triton import cdiv

from proteinclip import triton_kernels


def get_output_dtype(
    input_dtype: torch.dtype = torch.float32,
    autocast: typing.Optional[str] = None,
    ) -> torch.dtype:
    """
    Returns the appropriate output dtype for automatic mixed precision
    given the input dtype and the operation's autocast behaviour.

    Args:
        input_dtype: Input dtype.
        autocast: The relevent operation's autocast behaviour.
            None signifies the input dtype should flow through,
            'fp16' signifies autocasting to FP16 when AMP is enabled,
            and 'fp32' signifies autocasting to FP32 when AMP is enabled.
    """
    dtype = torch.get_autocast_dtype('cuda')
    assert dtype, \
        f'Only autocast to float16 is supported, received {dtype}'

    if torch.is_autocast_enabled():
        if autocast is None:
            return input_dtype

        elif autocast == 'fp16':
            return torch.float16

        elif autocast == 'fp32':
            return torch.float32

        else:
            raise RuntimeError(f'Autocast type {autocast} is invalid. '
                               'Options are None, fp16, and fp32')

    else:
        return input_dtype


class TritonLinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: typing.Any,
        inputs: torch.Tensor,
        weights: typing.Optional[torch.Tensor] = None,
        bias: typing.Optional[torch.Tensor] = None,
        act_func: typing.Optional[str] = None,  
    ) -> torch.Tensor:
        
        # IF actiavtion function is None, set it to 'gelu'
        if act_func is None:
            act_func = 'gelu'

        # If weights are None, throw an error
        if weights is None:
            raise ValueError("Weights must be provided")

        print(inputs.shape)

        flattened_inputs = inputs.flatten(0, -2)
        batch_dim, in_feat_dim = flattened_inputs.shape
        _, out_feat_dim = weights.shape

        # Create an empty torch tensor for the output
        outputs_dtype = get_output_dtype(inputs.dtype, autocast='fp16')
        outputs = torch.empty((batch_dim, out_feat_dim), dtype=outputs_dtype, device=inputs.device)

        # Launches a 1D grid, where each program outputs blocks of
        # BLOCK_SIZE_BATCH rows and BLOCK_SIZE_OUT_FEAT columns.
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']) *
                             cdiv(out_feat_dim, META['BLOCK_SIZE_OUT_FEAT']),)
        triton_kernels.triton_linear_forward_kernel[grid](
            flattened_inputs, 
            weights, 
            outputs,
            batch_dim, 
            in_feat_dim, 
            out_feat_dim,
            *flattened_inputs.stride(),
            *weights.stride(),
            *outputs.stride(),
            fp16=outputs_dtype is torch.float16
        )


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



