# Inspired from: https://github.com/BobMcDear/attorch/blob/main/attorch/act_kernels.py

import triton
import triton.language as tl

# TODO Write tests
@triton.jit
def gelu(input):
    """
    Applies GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input


@triton.jit
def gelu_grad(input):
    """
    Calculates the gradient of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    cdf_grad = 0.39894228 * tl.exp(-0.5 * input * input)
    return (cdf_grad * input + cdf)


@triton.jit
def apply_act_func(input, act_func: tl.constexpr):
    """
    Applies an activation function to the input.
    """
    if act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu(input)
    
    return output


@triton.jit
def apply_act_func_grad(output_grad, input, act_func):
    """
    Calculates the gradient of an activation function.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Gradient of the desired activation function.
    """

    if act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu_grad(input)

    return output_grad * output


# @triton.autotune(
#     configs=element_wise_kernel_configs(),
#     key=['size'],
# )
@triton.jit
def act_func_forward_kernel(
    input_pointer,
    output_pointer,
    act_func: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 32
):

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < input_pointer.shape[0]

    input = tl.load(input_pointer + offset, mask=mask)
    output = tl.save(output_pointer + offset, apply_act_func(input, act_func), mask=mask)


# @triton.autotune(
#     configs=element_wise_kernel_configs(),
#     key=['size'],
# )
@triton.jit
def act_func_backward_kernel(
    output_grad_pointer,
    input_pointer,
    input_grad_pointer,
    size,
    act_func: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 32

):
    """
    Triton kernel for activation function backward pass.
    """
    # Calculate the gradient of the activation function
    pid = tl.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input = tl.load(input_pointer + offset, mask=mask)
    input_grad = tl.store(input_grad_pointer + offset, apply_act_func_grad(output_grad, input, act_func), mask=mask)
