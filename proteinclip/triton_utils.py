# Inspired by https://github.com/BobMcDear/attorch/blob/main/attorch/utils.py

from typing import Dict, List
import typing
import torch
import triton
from triton import next_power_of_2


def BLOCK_SIZE_BATCH_heuristic(args: Dict) -> int:
    """
    Approximates an appropriate batch block size for softmax using a heuristic.

    Args:
        args: Arguments to softmax kernel.

    Returns:
        Appropriate batch block size.
    """
    # This heuristic was derived manually.
    # Essentially, if the batch dimension is greater than 1024,
    # for small feature sizes (less than 64), it is much more efficient
    # to process multiple rows at once in a given program.
    # Specifically, each time the number of samples is doubled,
    # the block size across the batch dimension should be doubled too,
    # with an upper bound of 128.
    return (min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128)
            if args['feat_dim'] < 64 else 1)


def allow_tf32() -> bool:
    """
    Returns whether the current GPU architecture supports TF32.
    """
    return torch.cuda.get_device_capability()[0] >= 8


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


def warps_kernel_configs() -> List[triton.Config]:
    """
    Returns kernel configurations with all possible number of warps.
    """
    return [triton.Config({}, num_warps=2**i) for i in range(6)]
