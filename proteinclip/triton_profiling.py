
from torch.utils.tensorboard import SummaryWriter
import torch

from proteinclip.triton_model import ContrastiveEmbeddingTriton, ContrastiveEmbeddingTorch

# Create random data
batch_size = 192
in_dim = 128
mlp_dim = 1024
shared_dim = 2048

tmp_x1 = torch.rand(batch_size, in_dim).to(torch.device('cuda'))
tmp_x2 = torch.rand(batch_size, in_dim).to(torch.device('cuda'))
tmp_batch = {"x_1": tmp_x1, "x_2": tmp_x2}

# Instantiate the model
triton_model = ContrastiveEmbeddingTriton(in_dim, in_dim, shared_dim).to(torch.device('cuda'))
torch_model = ContrastiveEmbeddingTorch(in_dim, in_dim, shared_dim).to(torch.device('cuda'))


# Tensorboard
writer = SummaryWriter(log_dir="/home/ubuntu/Krishna-Llama/tb_logs",
                        flush_secs=30)


def profile_model(model, tmp_batch, log_dir, num_iters=100):
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    prof.start()

    # Profile model forward pass with a context manager
    for iter in range(num_iters):
        with torch.no_grad():
            model_forward = model(tmp_batch)

        # send a signal to the profiler that the next iteration has started
        prof.step()
        
    prof.stop()


# Profile triton model
profile_model(triton_model, tmp_batch, "/home/ubuntu/Krishna-Llama/tb_logs/proteinclip/triton_contrastive_embedding", num_iters=100)

# Profile torch model
profile_model(torch_model, tmp_batch, "/home/ubuntu/Krishna-Llama/tb_logs/proteinclip/torch_contrastive_embedding", num_iters=100)
