############################################################################################################################
import os

# Triton debug options: Change to 1 to enable debugging and change to 0 to disable debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TRITON_INTERPRET'] = '1'
# This has to be before importing triton
############################################################################################################################


import json
import argparse
import logging

import torch
from torch import nn
from torch.utils import data

from proteinclip import data_utils

# from proteinclip import data_utils, fasta_utils, swissprot, hparams
from proteinclip import contrastive
from proteinclip import triton_layers


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
batch_size = 4
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
model_class = (
    contrastive.ContrastiveEmbeddingWithPreprocessor
    if do_per_token
    else contrastive.ContrastiveEmbedding
)
net = model_class(
    input_dim_1=input_dim_1,
    input_dim_2=input_dim_2,
    shared_dim=mlp_dim,
    num_hidden=mlp_n_hidden,
    lr=lr,
)
print("Defined network")


# Triton
sample_batch = next(iter(train_dl))
mlp_layer_1 = triton_layers.TritonLinearLayer(sample_batch["x_1"].shape[-1], sample_batch["x_1"].shape[-1], "gelu")
tmp_batch = sample_batch["x_1"].to(torch.device('cuda'))
mlp_layer_1_forward = mlp_layer_1(tmp_batch)



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

