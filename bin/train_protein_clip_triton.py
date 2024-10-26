############################################################################################################################
import os

# Triton debug options: Change to 1 to enable debugging and change to 0 to disable debugging
debug = False
if debug:
    os.environ['TRITON_INTERPRET'] = '1'
else:
    os.environ['TRITON_INTERPRET'] = '0'

# This has to be before importing triton
############################################################################################################################

import json
import argparse
import logging

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import triton

from proteinclip import data_utils

# from proteinclip import data_utils, fasta_utils, swissprot, hparams
from proteinclip import contrastive
from proteinclip import triton_layers, triton_layer_norm_layer
from proteinclip.test_matmul import triton_matmul

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def write_split_identifiers(train_ids, valid_ids, test_ids, out_file):
    """Write the data split identifiers to the given output .json."""
    with open(out_file, "w") as sink:
        json.dump(
            {
                "train": train_ids,
                "valid": valid_ids,
                "test": test_ids,
            },
            sink,
            indent=4,
        )


# Set random seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Training
# Local zenodo dir in the current file's grandparent directory
local_zenodo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "zenodo")
# Make zenodo dir if it does not exist
if not os.path.exists(local_zenodo_dir):
    os.makedirs(local_zenodo_dir)

local_text_path = "uniprot_sprot.dat.gz"
local_text_embed_path = "uniprot_text_embed.text-embedding-3-large.hdf5"
local_protein_embed_path = "esm_6layer_split0.hdf5"
# Download from zenodo if paths do not exist locally
## Change directory to the zenodo directory
os.chdir(local_zenodo_dir)
# Check if the local_text_path exists
if not os.path.exists(local_text_path):
    os.system(f"zenodo_get -g {local_text_path} 10.5281/zenodo.11176863")
# Check if the local_text_embed_path exists
if not os.path.exists(local_text_embed_path):
    os.system(f"zenodo_get -g {local_text_embed_path} 10.5281/zenodo.11176863")
# Check if the local_protein_embed_path exists
if not os.path.exists(local_protein_embed_path):
    os.system(f"zenodo_get -g {local_protein_embed_path} 10.5281/zenodo.11176863")

# If we want our own text embeddings, we can use the following code
# embed_write_json = os.path.join(local_zenodo_dir, f"text_embeddings_{text_embed_model}.json")
# sp_text_embed = swissprot.embed_function_descriptions(local_text_path, model=text_embed_model, write_json=embed_write_json)

print("Downloaded data from zenodo")
# Change directory back to the current file's directory
os.chdir(os.path.dirname(__file__))
print("Changed directory back to the current file's directory")


# # Read in the training config
# hyperparameters = hparams.read_hparams(args.training_config)
# logging.info(f"Hyperparameters: {hyperparameters}")

# Load precomputed ESM2 embeddings
local_protein_embed_path = [os.path.join(local_zenodo_dir, local_protein_embed_path)]
esm_embeddings = data_utils.MultiH5(local_protein_embed_path)
print(esm_embeddings.keys())

# Load in the precomputed GPT text embeddings
local_text_embed_path = [os.path.join(local_zenodo_dir, local_text_embed_path)]
sp_text_embed = data_utils.MultiH5(local_text_embed_path)
print(sp_text_embed.keys())


print("Done loading embeddings")

# # Identify shared keys
shared_keys = sorted(set(esm_embeddings.mapping.keys()).intersection(sp_text_embed.mapping.keys()))
print(f"Number of shared keys: {len(shared_keys)}")

# Subset some pairs randomly for debugging
if debug:
    shared_keys = shared_keys[:1000]

do_per_token = False
do_unit_norm = True
# Create dataset; first item is ESM, second is text
if do_per_token:
    dset = data_utils.CLIPDataset2D1D(
        pairs=shared_keys, map1=esm_embeddings, map2=sp_text_embed
    )
else:
    dset = data_utils.CLIPDataset(
        pairs=shared_keys,
        map1=esm_embeddings,
        map2=sp_text_embed,
        enforce_unit_norm=do_unit_norm,
    )


# Create data splits
# For now just do random
train_splitfile = None
if not train_splitfile:
    split_indices = data_utils.random_split(len(dset), [0.9, 0.05, 0.05])
    logging.info(f"Randomized split sizes: {[len(x) for x in split_indices]}")
else:
    assert os.path.isfile(train_splitfile)
    with open(train_splitfile, "r") as source:
        splits = json.load(source)
    assert "valid" in splits and "test" in splits
    if "train" not in splits:
        logging.warning(
            "No 'train' in splits; using all non-valid/test pairs to train."
        )
        splits["train"] = [
            p
            for p in dset.pairs
            if p not in splits["valid"] and p not in splits["test"]
        ]
    logging.info(
        f"Loaded split IDs: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}"
    )
    # Pre-cache the mapping for identifier -> index
    id2index = {p: i for i, p in enumerate(dset.pairs)}
    split_indices = [
        [id2index[p] for p in splits[s] if p in id2index]
        for s in ("train", "valid", "test")
    ]
dset_splits = [data.Subset(dset, idx) for idx in split_indices]

# Create data loaders
batch_size = 192
train_dl, valid_dl, _test_dl = [
    data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(i == 0),
        # drop_last=(i == 0),
        num_workers=8,
        pin_memory=True,
    )
    for i, ds in enumerate(dset_splits)
]
print("Created data loaders")

# Definte network
mlp_dim = 256
mlp_n_hidden = 1
lr = 1e-4
input_dim_1 = next(iter(train_dl))["x_1"].shape[-1]
input_dim_2 = next(iter(train_dl))["x_2"].shape[-1]
# model_class = (
#     contrastive.ContrastiveEmbeddingWithPreprocessor
#     if do_per_token
#     else contrastive.ContrastiveEmbedding
# )
# net = model_class(
#     input_dim_1=input_dim_1,
#     input_dim_2=input_dim_2,
#     shared_dim=mlp_dim,
#     num_hidden=mlp_n_hidden,
#     lr=lr,
# )
print("Defined network")


# Create a model using Triton
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


# Instantiate the model
custom_net = ContrastiveEmbeddingTriton(input_dim_1, input_dim_2, mlp_dim)


# Train loop over epochs and batches
lr = 0.001
optimizer = torch.optim.Adam(custom_net.parameters(), lr=lr)
checkpoint_dir = "/home/ubuntu/Krishna-Llama/proteinclip/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
num_epochs = 10
for epoch in range(num_epochs):
    # Go over train_dl
    for batch_index, batch in enumerate(tqdm(train_dl, desc="Training Epoch")):

        # Forward pass
        batch = {k: v.to(torch.device('cuda')) for k, v in batch.items()}
        x1_proj, x2_proj = custom_net(batch)
        
        # Loss
        optimizer.zero_grad()
        temperature = nn.Parameter(data=torch.Tensor([1.0]), requires_grad=True).to(x1_proj.device)
        logits = x1_proj @ x2_proj.T * torch.exp(temperature)
        labels = torch.arange(x1_proj.shape[0]).to(logits.device)
        l_1 = F.cross_entropy(logits, labels)
        l_2 = F.cross_entropy(logits.T, labels)
        loss = (l_1 + l_2) / 2

        # Backward pass and step
        loss.backward()
        optimizer.step()

        # Log loss
        if batch_index % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_index}, Loss: {loss.item()}")
        
    # Save checkpoint
    torch.save(custom_net.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt"))
    



# Testing backward
loss = torch.sum(mlp_layer_1_shared_forward)
loss.backward()
# Check the gradients
print(tmp_batch.grad)
print(mlp_layer_1.weight.grad)

# Test if the forward pass matches using torch
tm = torch.matmul(tmp_batch, mlp_layer_1.weight)
torch.allclose(mlp_layer_1_forward, tm, rtol=1e-2, atol=1e-2)


# Layer comparison
mlp_torch_layer_1 = nn.Linear(sample_batch["x_1"].shape[-1], sample_batch["x_1"].shape[-1]).to(torch.device('cuda'))
mlp_torch_layer_1.weight = nn.Parameter(mlp_layer_1.weight.t())
mlp_torch_layer_1.bias = None
mlp_torch_layer_1_forward = mlp_torch_layer_1(tmp_batch)
# Gelu
mlp_torch_layer_1_forward = nn.functional.gelu(mlp_torch_layer_1_forward)
# Check
assert torch.allclose(mlp_layer_1_forward, mlp_torch_layer_1_forward, rtol=1e-2, atol=1e-2)

# Shared layer
mlp_torch_layer_1_shared = nn.Linear(sample_batch["x_1"].shape[-1], mlp_dim).to(torch.device('cuda'))
mlp_torch_layer_1_shared.weight = nn.Parameter(mlp_layer_1_shared.weight.t())
mlp_torch_layer_1_shared.bias = None
mlp_torch_layer_1_shared_forward = mlp_torch_layer_1_shared(mlp_torch_layer_1_forward)
# mlp_torch_layer_1_shared_forward = nn.functional.gelu(mlp_torch_layer_1_shared_forward)
# Check
assert torch.allclose(mlp_layer_1_shared_forward, mlp_torch_layer_1_shared_forward, rtol=1e-2, atol=1e-2)

torch_loss = torch.sum(mlp_torch_layer_1_shared_forward)
torch_loss.backward()

# Check the gradients
assert torch.allclose(mlp_torch_layer_1_shared.weight.grad.t(), mlp_layer_1_shared.weight.grad, rtol=1e-2, atol=1e-2)
assert torch.allclose(mlp_torch_layer_1.weight.grad.t(), mlp_layer_1.weight.grad, rtol=1e-2, atol=1e-2)



# Tensorboard graph
writer = SummaryWriter(log_dir="/home/ubuntu/Krishna-Llama/tb_logs",
                            flush_secs=30)
writer.add_graph(mlp_layer_1,tmp_batch.half())

# Benchmark
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["batch_size"],  # Argument names to use as an x-axis for the plot
        x_vals=[16 * i for i in range(1, 10)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["cublas", "triton"],  # Label name for the lines
        line_names=["cuBLAS", "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp16",
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(batch_size, provider):
    train_dl, valid_dl, _test_dl = [
        data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(i == 0),
            # drop_last=(i == 0),
            num_workers=8,
            pin_memory=True,
        )
        for i, ds in enumerate(dset_splits)
    ]
    sample_batch = next(iter(train_dl))
    tmp_batch = sample_batch["x_1"].to(torch.device('cuda'))
    mlp_triton_layer_1 = triton_layers.TritonLinearLayer(sample_batch["x_1"].shape[-1], sample_batch["x_1"].shape[-1], "gelu")
    mlp_torch_layer_1 = nn.Linear(sample_batch["x_1"].shape[-1], sample_batch["x_1"].shape[-1]).to(torch.device('cuda'))

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda:mlp_torch_layer_1(tmp_batch), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: mlp_triton_layer_1(tmp_batch), quantiles=quantiles)
    perf = lambda ms: 2 * batch_size*320*320 * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


bench_out = benchmark.run(show_plots=True, print_data=True)
# Save plots
benchmark.save_all_plots("/home/ubuntu/Krishna-Llama/triton_benchmarks")


do_profile = False
if do_profile:
    # Torch
    writer = SummaryWriter(log_dir="/home/ubuntu/Krishna-Llama/tb_logs",
                            flush_secs=30)

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
        # on_trace_ready=partial(trace_handler,
        #                        results_dir="./profiler_logs"),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/ubuntu/Krishna-Llama/tb_logs/triton_log_norm"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    prof.start()

    for iter in range(100):
        with torch.no_grad():
            mlp_layer_1_forward = mlp_layer_1(tmp_batch)
        # send a signal to the profiler that the next iteration has started
        prof.step()
        
    prof.stop()
# # Define logger, write configuration files and data splits
# logger = CSVLogger(save_dir=args.out, name=args.name)
# logger.log_hyperparams(hyperparameters.as_dict())
# write_split_identifiers(
#     train_ids=[dset.pairs[i] for i in split_indices[0]],
#     valid_ids=[dset.pairs[i] for i in split_indices[1]],
#     test_ids=[dset.pairs[i] for i in split_indices[2]],
#     out_file=os.path.join(logger.log_dir, "data_splits.json"),
# )
# net.write_config_json(os.path.join(logger.log_dir, "model_config.json"))
# with open(os.path.join(logger.log_dir, "training_config.json"), "w") as sink:
#     json.dump(vars(args), sink, indent=4)

# # Train
# trainer = pl.Trainer(
#     max_epochs=hyperparameters.max_epochs,
#     accelerator="cuda",
#     devices=args.gpu,
#     enable_progress_bar=True,
#     logger=logger,
#     log_every_n_steps=10,
#     deterministic=True,
# )
# trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)

# # Export model as ONNX files
# contrastive.model_to_onnx(
#     net.project_1,
#     os.path.join(logger.log_dir, "project_1.onnx"),
#     input_shape=(input_dim_1,),
# )
# contrastive.model_to_onnx(
#     net.project_2,
#     os.path.join(logger.log_dir, "project_2.onnx"),
#     input_shape=(input_dim_2,),
# )

