
import torch
import triton
import triton.language as tl
import pdb


def allow_tf32() -> bool:
    """
    Returns whether the current GPU architecture supports TF32.
    """
    return torch.cuda.get_device_capability()[0] >= 8


def get_n_stages(n_stages: int = 2) -> int:
    """
    Receives number of stages for software pipelining and returns it as-is
    if the GPU architecture is Ampere or newer and 2 otherwise.
    """
    return 2 if torch.cuda.get_device_capability()[0] < 8 else n_stages


def linear_forward_config(
    BLOCK_SIZE_BATCH: int,
    BLOCK_SIZE_IN_FEAT: int,
    BLOCK_SIZE_OUT_FEAT: int,
    GROUP_SIZE_BATCH: int = 8,
    n_warps: int = 4,
    n_stages: int = 2,
    ) -> triton.Config:
    """
    Creates a triton.Config object for linear_forward_kernel
    given meta-parameters for auto-tuning.

    Args:
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
        GROUP_SIZE_BATCH: Group size across the batch dimension.
        n_warps: Number of warps to use for the kernel when compiled for GPUs.
        n_stages: Number of stages the compiler uses to software-pipeline.
            On GPU architectures older than Ampere, this is fixed at 2.

    Returns:
        Kernel configuration.
    """
    return triton.Config({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH,
                          'BLOCK_SIZE_IN_FEAT': BLOCK_SIZE_IN_FEAT,
                          'BLOCK_SIZE_OUT_FEAT': BLOCK_SIZE_OUT_FEAT,
                          'GROUP_SIZE_BATCH': GROUP_SIZE_BATCH},
                          num_warps=n_warps, num_stages=get_n_stages(n_stages))


@triton.autotune(
    configs=[
        linear_forward_config(4, 32, 32, n_warps=2, n_stages=2),
        linear_forward_config(32, 32, 32, n_warps=2, n_stages=2),
        linear_forward_config(64, 32, 32, n_warps=2, n_stages=5),
        linear_forward_config(64, 32, 128, n_warps=4, n_stages=4),
        linear_forward_config(64, 32, 256, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 32, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 64, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 128, n_warps=4, n_stages=4),
        linear_forward_config(128, 64, 256, n_warps=8, n_stages=3),
    ],
    key=['batch_dim', 'in_feat_dim', 'out_feat_dim', 'fp16'],
)
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def triton_linear_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    batch_dim,
    in_feat_dim,
    out_feat_dim,
    fp16: tl.constexpr, 
    tf32: tl.constexpr,
    BLOCK_SIZE_BATCH:  tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    GROUP_SIZE_BATCH: tl.constexpr,
):
    # Print the loaded values
    tl.device_print("Loaded values: ")
    pid = tl.program_id(0)
    n_batch_pids = tl.cdiv(batch_dim, BLOCK_SIZE_BATCH)
    n_out_feat_pids = tl.cdiv(out_feat_dim, BLOCK_SIZE_OUT_FEAT)
    pids_per_group = GROUP_SIZE_BATCH * n_out_feat_pids
    group_id = pid // pids_per_group
