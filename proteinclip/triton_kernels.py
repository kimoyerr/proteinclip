import numpy as np
import triton
import triton.language as tl
import torch


def allow_tf32() -> bool:
    """
    Returns whether the current GPU architecture supports TF32.
    """
    # return torch.cuda.get_device_capability()[0] >= 8
    return False


# def get_n_stages(n_stages: int = 2) -> int:
#     """
#     Receives number of stages for software pipelining and returns it as-is
#     if the GPU architecture is Ampere or newer and 2 otherwise.
#     """
#     return 2 if torch.cuda.get_device_capability()[0] < 8 else n_stages


# def linear_forward_config(
#     BLOCK_SIZE_BATCH: int,
#     BLOCK_SIZE_IN_FEAT: int,
#     BLOCK_SIZE_OUT_FEAT: int,
#     GROUP_SIZE_BATCH: int = 8,
#     n_warps: int = 4,
#     n_stages: int = 2,
#     ) -> triton.Config:
#     """
#     Creates a triton.Config object for linear_forward_kernel
#     given meta-parameters for auto-tuning.

#     Args:
#         BLOCK_SIZE_BATCH: Block size across the batch dimension.
#         BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
#         BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
#         GROUP_SIZE_BATCH: Group size across the batch dimension.
#         n_warps: Number of warps to use for the kernel when compiled for GPUs.
#         n_stages: Number of stages the compiler uses to software-pipeline.
#             On GPU architectures older than Ampere, this is fixed at 2.

#     Returns:
#         Kernel configuration.
#     """
#     return triton.Config({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH,
#                           'BLOCK_SIZE_IN_FEAT': BLOCK_SIZE_IN_FEAT,
#                           'BLOCK_SIZE_OUT_FEAT': BLOCK_SIZE_OUT_FEAT,
#                           'GROUP_SIZE_BATCH': GROUP_SIZE_BATCH},
#                           num_warps=n_warps, num_stages=get_n_stages(n_stages))


# @triton.autotune(
#     configs=[
#         linear_forward_config(4, 32, 32, n_warps=2, n_stages=2),
#         linear_forward_config(32, 32, 32, n_warps=2, n_stages=2),
#         linear_forward_config(64, 32, 32, n_warps=2, n_stages=5),
#         linear_forward_config(64, 32, 128, n_warps=4, n_stages=4),
#         linear_forward_config(64, 32, 256, n_warps=4, n_stages=4),
#         linear_forward_config(128, 32, 32, n_warps=4, n_stages=4),
#         linear_forward_config(128, 32, 64, n_warps=4, n_stages=4),
#         linear_forward_config(128, 32, 128, n_warps=4, n_stages=4),
#         linear_forward_config(128, 64, 256, n_warps=8, n_stages=3),
#     ],
#     key=['batch_dim', 'in_feat_dim', 'out_feat_dim', 'fp16'],
# )
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def triton_linear_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    batch_dim,
    in_feat_dim,
    out_feat_dim,  
    input_batch_stride,
    input_in_feat_stride,
    weight_in_feat_stride,
    weight_out_feat_stride,
    output_batch_stride,
    output_out_feat_stride, 
    fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH:  tl.constexpr = 16,  # TODO: Change this in prod to use triton.autotune configs above that have been commented out
    BLOCK_SIZE_IN_FEAT: tl.constexpr = 64,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr = 32,
    GROUP_SIZE_BATCH: tl.constexpr = 8,
    # Best for 320 input features
    # BLOCK_SIZE_BATCH:  tl.constexpr = 128,  # TODO: Change this in prod to use triton.autotune configs above that have been commented out
    # BLOCK_SIZE_IN_FEAT: tl.constexpr = 64,
    # BLOCK_SIZE_OUT_FEAT: tl.constexpr = 128,
    # GROUP_SIZE_BATCH: tl.constexpr = 32,
):
    # Print the loaded values
    # tl.device_print("Loaded values: ")
    pid = tl.program_id(0)
    n_batch_pids = tl.cdiv(batch_dim, BLOCK_SIZE_BATCH)
    n_out_feat_pids = tl.cdiv(out_feat_dim, BLOCK_SIZE_OUT_FEAT)
    
    # This code chunk groups batches and output features together for the tiled matrix multiplication: https://alvinwan.com/how-to-tile-matrix-multiplication
    ## The batch_dimension moves faster than the output feature dimension
    pids_per_group = GROUP_SIZE_BATCH * n_out_feat_pids  # Number of pids needed to cover all output features in a group. Example: If the output features are 128 and BLOCK_SIZE_OUT_FEAT is 32, and GROUP_SIZE_BATCH 4 is  then pids_per_group = 4*4 = 16
    group_id = pid // pids_per_group  # Group id for the current pid
    first_batch_pid = group_id * GROUP_SIZE_BATCH
    GROUP_SIZE_OUT_FEAT = min(GROUP_SIZE_BATCH, n_batch_pids - first_batch_pid)  # Last (or the only) group may have fewer pids than GROUP_SIZE_BATCH
    if GROUP_SIZE_OUT_FEAT < 0:
        print("Triton Kernel: Done")
    batch_pid = first_batch_pid + (pid % pids_per_group) % GROUP_SIZE_OUT_FEAT  # Move to the next batch when all output features are covered
    out_feat_pid = (pid % pids_per_group) // GROUP_SIZE_OUT_FEAT  # Move to the next group of output features when all batches are covered

    batch_offset = batch_pid*BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    out_feat_offset = out_feat_pid*BLOCK_SIZE_OUT_FEAT + tl.arange(0, BLOCK_SIZE_OUT_FEAT)

    # Guide for the offsets
    batch_mask = batch_offset < batch_dim
    out_feat_mask = out_feat_offset < out_feat_dim

    input_pointer += input_batch_stride*batch_offset[:, None]
    weight_pointer += weight_out_feat_stride*out_feat_offset[None, :]


    # Run inner dimension loop for each block
    accum = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_FEAT),
                     dtype=tl.float32)
    for block_ind in range(0, tl.cdiv(in_feat_dim, BLOCK_SIZE_IN_FEAT)):
        in_feat_offset = (block_ind * BLOCK_SIZE_IN_FEAT +
                          tl.arange(0, BLOCK_SIZE_IN_FEAT))
        in_feat_mask = in_feat_offset < in_feat_dim

        curr_input_pointer = (input_pointer +
                              input_in_feat_stride * in_feat_offset[None, :])
        curr_weight_pointer = (weight_pointer +
                               weight_in_feat_stride * in_feat_offset[:, None])

        # Load the input and weight blocks
        input_block = tl.load(curr_input_pointer,
                              mask=batch_mask[:, None] & in_feat_mask[None, :])
        weight_block = tl.load(curr_weight_pointer,
                               mask=out_feat_mask[None, :] & in_feat_mask[:, None])

        if fp16:
            input_block = input_block.to(tl.float16)
            weight_block = weight_block.to(tl.float16)

        accum += tl.dot(input_block, weight_block, allow_tf32=tf32)

    # if add_bias:
    #     bias = tl.load(bias_pointer + out_feat_offset,
    #                    mask=out_feat_mask)

    #     if fp16:
    #         bias = bias.to(tl.float16)

    #     accum += bias[None, :]

    # if act_func is not None:
    #     if save_pre_act:
    #         pre_act_pointer += (pre_act_batch_stride * batch_offset[:, None] +
    #                             pre_act_out_feat_stride * out_feat_offset[None, :])
    #         tl.store(pre_act_pointer, accum,
    #                  mask=batch_mask[:, None] & out_feat_mask[None, :])

    #     accum = apply_act_func(accum, None, None, None, param, act_func, False)

    output_pointer += (output_batch_stride * batch_offset[:, None] +
                       output_out_feat_stride * out_feat_offset[None, :])
    tl.store(output_pointer, accum,
             mask=batch_mask[:, None] & out_feat_mask[None, :])

    print("Triton Kernel: Done")




