#python
#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Script: scGPT Fine-tuning with Hierarchical Cell-Type Annotation
# Description: 
#   This script fine-tunes the pre-trained scGPT model for cell-type annotation.
#   It features custom preprocessing (encoding fixes, deduplication), 
#   hierarchy-aware loss functions (e.g., hierarchical cross-entropy), 
#   and depth-wise hierarchical evaluation metrics.
# ==============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import pandas as pd
import anndata as ad
from anytree import Node, RenderTree
from anytree.search import find
from collections import defaultdict
import unicodedata
from collections import Counter

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics


sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


# ## Step1: Specify hyper-parameter setup for cell-type annotation task
# Listed below are some hyper-parameter recommendations for the cell-type task. Note that the CLS objective is on to facilitate cell-type classification.
# ------------------------------------------------------------------------------
# Hyperparameter Configuration
# ------------------------------------------------------------------------------
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=True,
    load_model="/home/mengjiayue/sci_data/save/scGPT_human/",
    mask_ratio=0.0,
    epochs=10,
    n_bins=51,
    MVC=False, # Masked value prediction for cell embedding
    ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = False,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
)


if os.environ.get("DATASET_NAME"):
    hyperparameter_defaults["dataset_name"] = os.environ["DATASET_NAME"]


run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
    mode="disabled",                 # Add this line to completely disable wandb
)



config = wandb.config
print(config)
set_seed(config.seed)
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 3001
n_bins = config.n_bins
# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight
explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training
per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1
# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability
# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True



# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")


DAB_separate_optim = True if DAB > 1 else False
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
# ## Step 2: Load and pre-process data
# We follow the standard scGPT data pre-processing pipelines for the cell-type annotation task. Note that since now we have two datasets at hand (i.e., reference and query data), the same pre-prpocessing steps need to be applied to both of them.
##################################################################################################################
if dataset_name != "xxxxxxxxxxxxxxxxxxxxxxxx":
    data_dir = Path("./")
    adata_train = sc.read(data_dir / "raw.train.h5ad")
    adata_valid = sc.read(data_dir / "raw.valid.h5ad")
    adata_test = sc.read(data_dir / "raw.test.h5ad")
    adata_train.X = adata_train.layers['raw_counts']   
    adata_valid.X = adata_valid.layers['raw_counts']
    adata_test.X = adata_test.layers['raw_counts']
    # ============================================================
    # Block 1. Encoding scan + fix + per-split label distribution
    # (runs on train / valid / test independently)
    # ============================================================
    SPLITS = ["train", "valid", "test"]
    # --- 1a. Non-ASCII characters per split -----------------------------------
    print("\n" + "=" * 70)
    print("1a. Non-ASCII characters present in `cell_type` (per split)")
    print("=" * 70)
    for nm in SPLITS:
        ad = globals()[f"adata_{nm}"]
        cnt = Counter()
        for s in ad.obs["cell_type"].astype(str).unique():
            for ch in s:
                if ord(ch) > 127:
                    cnt[repr(ch)] += 1
        print(f"[{nm}] n_types={ad.obs['cell_type'].nunique():3d}  non-ASCII={dict(cnt)}")
    # --- 1b. Suspect corrupted labels per split -------------------------------
    print("\n1b. Suspect encoding-corrupted labels (contain '\\u672a', '&#', '?')")
    for nm in SPLITS:
        ad = globals()[f"adata_{nm}"]
        bad = [s for s in ad.obs["cell_type"].astype(str).unique()
            if "\u672a" in s or "&#" in s or "?" in s]
        print(f"[{nm}] {len(bad)} suspects")
        for s in bad:
            n = int((ad.obs["cell_type"].astype(str) == s).sum())
            print(f"    n={n:5d}  {s!r}")
    # --- 1c. Fix encoding (per split) ----------------------------------------
    def _fix_encoding(s):
        """Recover UTF-8 chars mis-decoded as GBK (delta '\u03b4' shows up as '\u672a')."""
        if not isinstance(s, str):
            return s
        return (s.replace("V\u672a2", "V\u03b42")
                .replace("V\u672a1", "V\u03b41")
                .replace("\u672a", "\u03b4"))
    print("\n1c. Applying encoding fix")
    for nm in SPLITS:
        ad = globals()[f"adata_{nm}"]
        before = ad.obs["cell_type"].nunique()
        ad.obs["cell_type"] = ad.obs["cell_type"].astype(str).map(_fix_encoding)
        after = ad.obs["cell_type"].nunique()
        globals()[f"adata_{nm}"] = ad
        print(f"[{nm}] n_types {before} -> {after}"
            + ("  (merged!)" if after < before else ""))
    # --- 1d. Per-split label distribution ------------------------------------
    print("\n1d. Per-split label distribution (after encoding fix)")
    for nm in SPLITS:
        ad = globals()[f"adata_{nm}"]
        vc = ad.obs["cell_type"].value_counts()
        print(f"\n[{nm}] n_obs={ad.n_obs}, n_types={len(vc)}, "
            f"min={vc.min()}, median={int(vc.median())}, max={vc.max()}")
        print(f"  rarest 8:")
        for ct, n in vc.tail(8).items():
            print(f"    n={n:5d}  {ct!r}")
        print(f"  top 8:")
        for ct, n in vc.head(8).items():
            print(f"    n={n:5d}  {ct!r}")
    # --- 1e. Cross-split label presence --------------------------------------
    print("\n1e. Cross-split label presence")
    s = {nm: set(globals()[f"adata_{nm}"].obs["cell_type"].astype(str).unique())
        for nm in SPLITS}
    only_tr = s["train"] - s["valid"] - s["test"]
    only_va = s["valid"] - s["train"] - s["test"]
    only_te = s["test"]  - s["train"] - s["valid"]
    tr_not_va = s["train"] - s["valid"]
    tr_not_te = s["train"] - s["test"]
    va_not_te = s["valid"] - s["test"]
    print(f"  in ALL three         : {len(s['train'] & s['valid'] & s['test'])}")
    print(f"  only in train        : {len(only_tr)}")
    print(f"  only in valid        : {len(only_va)}")
    print(f"  only in test         : {len(only_te)}")
    print(f"  train - valid        : {len(tr_not_va)}")
    print(f"  train - test         : {len(tr_not_te)}")
    print(f"  valid - test         : {len(va_not_te)}")
    if only_te:
        n_unseen = int(globals()["adata_test"].obs["cell_type"].astype(str).isin(only_te).sum())
        pct = n_unseen / max(1, len(globals()["adata_test"]))
        print(f"  -> test cells whose label is absent from train AND valid: "
            f"{n_unseen} ({pct*100:.2f}%)   [guaranteed 0 recall]")
        for ct in sorted(only_te):
            n = int((globals()["adata_test"].obs["cell_type"].astype(str) == ct).sum())
            print(f"     n={n:4d}  {ct!r}")
    if only_va:
        n_unseen = int(globals()["adata_valid"].obs["cell_type"].astype(str).isin(only_va).sum())
        pct = n_unseen / max(1, len(globals()["adata_valid"]))
        print(f"  -> valid cells whose label is absent from train AND test: "
            f"{n_unseen} ({pct*100:.2f}%)")
    print("=" * 70 + "\n")
    ##################
    adata_train.var_names = adata_train.var["original_index"].astype(str).values
    adata_valid.var_names = adata_valid.var["original_index"].astype(str).values
    adata_test.var_names = adata_test.var["original_index"].astype(str).values
    adata_train.obs["celltype"] = adata_train.obs["cell_type"].astype("category")
    adata_valid.obs["celltype"] = adata_valid.obs["cell_type"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
    adata_train.obs["batch_id"]  = adata_train.obs["str_batch"] = "0"
    adata_valid.obs["batch_id"]  = adata_valid.obs["str_batch"] = "1"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "2"
    #adata.var.set_index(adata.var["gene_name"], inplace=True)
    #adata_test.var.set_index(adata.var["gene_name"], inplace=True)
    data_is_raw = True
    filter_gene_by_counts = False
    ###############
     # ==============================================================================
    # Section 1: Data Loading, Encoding Fix, and Deduplication
    # ==============================================================================
    def _bkey(ad):
        if "cell_name" in ad.obs.columns:
            return ad.obs["cell_name"].astype(str) + "|" + ad.obs_names.astype(str)
        return ad.obs_names.astype(str)

    # ---- 1)  ----
    for nm in ["train", "valid", "test"]:
        ad = locals()[f"adata_{nm}"]
        k = _bkey(ad)
        dup = k.duplicated(keep="first").values
        print(f"[{nm}] internal dup barcodes: {int(dup.sum())}  ({ad.n_obs} -> {int((~dup).sum())})")
        ad = ad[~dup].copy()
        globals()[f"adata_{nm}"] = ad

    # ---- 2) ----
    train_keys = set(_bkey(adata_train))

    mask_v = _bkey(adata_valid).isin(train_keys)
    n_v = int(mask_v.sum())
    removed_valid = adata_valid[mask_v].copy()
    removed_valid.obs.to_csv(save_dir / "removed_valid_overlap_with_train.csv")
    print(f"[valid] overlap with train: {n_v}  ({adata_valid.n_obs} -> {adata_valid.n_obs - n_v})")
    adata_valid = adata_valid[~mask_v].copy()

    seen = train_keys | set(_bkey(adata_valid))
    mask_t = _bkey(adata_test).isin(seen)
    n_t = int(mask_t.sum())
    removed_test = adata_test[mask_t].copy()
    removed_test.obs.to_csv(save_dir / "removed_test_overlap_with_train_valid.csv")
    print(f"[test] overlap with train/valid: {n_t}  ({adata_test.n_obs} -> {adata_test.n_obs - n_t})")
    adata_test = adata_test[~mask_t].copy()

    # ---- 3)  ----
    print("\n========== Duplicate summary ==========")
    print(f"valid∩train = {n_v}, test∩(train∪valid) = {n_t}")
    print(f"Final: train={adata_train.n_obs}, valid={adata_valid.n_obs}, test={adata_test.n_obs}")
    if adata_valid.n_obs == 0:
        print("ERROR: valid empty after dedup. Check raw.valid.h5ad == raw.train.h5ad ?")
        sys.exit(1)
    if adata_test.n_obs == 0:
        print("ERROR: test empty after dedup. Check raw.test.h5ad overlaps with train.")
        sys.exit(1)
    print("========================================\n")
    #---------------------------------------
    adata_test_raw = adata_test.copy()
    for nm in ["train", "valid", "test"]:
        ad = globals()[f"adata_{nm}"]
        n_dup = int(ad.var_names.duplicated().sum())
        if n_dup > 0:
            examples = ad.var_names[ad.var_names.duplicated(keep=False)].unique()[:5].tolist()
            print(f"[{nm}] var_names duplicates: {n_dup}, examples: {examples}")
            ad.var_names_make_unique()
            print(f"[{nm}] after make_unique: n_vars={ad.n_vars}, "
                f"var_unique={ad.var_names.is_unique}")
        else:
            print(f"[{nm}] var_names already unique (n_vars={ad.n_vars})")
        assert ad.var_names.is_unique, f"{nm}: var_names still not unique!"
        globals()[f"adata_{nm}"] = ad
    #------------------------------------
    adata = adata_train.concatenate(adata_valid,adata_test, batch_key="str_batch")
    # --------------------------------------------------------------------------
    # Preserve the raw data 'adata' for downstream hierarchical evaluation.
    # 
    # Why we need this:
    #   1. scGPT's preprocessing steps (e.g., vocabulary filtering, subsetting 
    #      to highly variable genes) can remove certain genes or alter the 
    #      AnnData object's original structure.
    #   2. The `depth_accuracy_from_adata` function relies on the complete, 
    #      unfiltered `level_sequence` for every single cell in the set.
    #   3. Keeping an untouched copy ensures exact index alignment when we map 
    #      the model's predictions back to their true biological hierarchy.
    # --------------------------------------------------------------------------
# make the batch category column
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# ==============================================================================
# Section 2: Hierarchical Prior & Distance Matrix Construction
# ==============================================================================
# Parse the 'level_sequence' from AnnData to build the cell lineage tree.
# Compute the symmetric shortest-path distance between all cell types based on LCA (Lowest Common Ancestor).
###############################################################################################
# Get all unique cell types (leaf nodes)
unique_types = [id2type[i] for i in range(num_types)]
#unique_types = adata.obs['celltype'].unique()
#type_to_idx = {t: i for i, t in enumerate(unique_types)}
n_classes = len(unique_types)

# Store node list from root to leaf for each type
type_to_nodes = {}
for t in unique_types:
    seq = adata.obs[adata.obs['celltype'] == t]['level_sequence'].iloc[0]
    if pd.isna(seq):
        raise ValueError(f"Missing level_sequence for celltype {t}")
    nodes = [node.strip() for node in seq.split('->')]
    type_to_nodes[t] = ['ROOT'] + nodes

# Compute symmetric distance matrix
dist_mat = torch.zeros((n_classes, n_classes), dtype=torch.float)

for i, t_i in enumerate(unique_types):
    nodes_i = type_to_nodes[t_i]
    for j, t_j in enumerate(unique_types):
        if i == j:
            dist_mat[i, j] = 0
            continue
        nodes_j = type_to_nodes[t_j]
        # Find deepest common ancestor depth (starting from root)
        common_depth = -1
        min_len = min(len(nodes_i), len(nodes_j))
        for k in range(min_len):
            if nodes_i[k] == nodes_j[k]:
                common_depth = k
            else:
                break
        # Distance = (depth_i - common_depth) + (depth_j - common_depth)
        depth_i = len(nodes_i) - 1
        depth_j = len(nodes_j) - 1
        dist = (depth_i - common_depth) + (depth_j - common_depth)
        dist_mat[i, j] = dist

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dist_mat = dist_mat.to(device)
################################################################################################
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]


data_is_raw = True
# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=True,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
adata_train = adata[adata.obs["str_batch"] == "0"].copy()
adata_valid = adata[adata.obs["str_batch"] == "1"].copy()
adata_test  = adata[adata.obs["str_batch"] == "2"].copy()

preprocessor(adata_train, batch_key=None)
preprocessor(adata_valid, batch_key=None)
preprocessor(adata_test,  batch_key=None)


input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]


genes = adata_train.var["gene_name"].tolist()

# ---- Training Data Set----
all_counts_train = (
    adata_train.layers[input_layer_key].toarray()
    if issparse(adata_train.layers[input_layer_key])
    else adata_train.layers[input_layer_key]
)
train_celltype_labels = np.array(adata_train.obs["celltype_id"].tolist())
train_batch_labels    = np.array(adata_train.obs["batch_id"].tolist())

#-------------------------
# ============================================================
# MAPPING CHECK 2: all_counts_train row i <-> train_celltype_labels[i]
# ============================================================
print("\n" + "=" * 70)
print("MAP CHECK 2: all_counts_train vs train_celltype_labels")
print("=" * 70)
print(f"all_counts_train.shape = {all_counts_train.shape}")
print(f"train_celltype_labels.shape = {train_celltype_labels.shape}")
print(f"adata_train.layers['X_binned'].shape = {adata_train.layers['X_binned'].shape}")
print(f"adata_train.obs.shape = {adata_train.obs.shape}")

# row count must match
assert all_counts_train.shape[0] == len(train_celltype_labels), \
    f"ROW COUNT MISMATCH: {all_counts_train.shape[0]} vs {len(train_celltype_labels)}"
assert all_counts_train.shape[0] == adata_train.n_obs, \
    f"ROW COUNT MISMATCH: {all_counts_train.shape[0]} vs {adata_train.n_obs}"

# spot-check: for 5 random cells, compare n_nonzero in X_binned row
# with a recomputation from adata_train, plus verify label
import random
random.seed(0)
indices = random.sample(range(len(train_celltype_labels)), 5)
for i in indices:
    X_row = all_counts_train[i]
    n_nonzero = int((X_row > 0).sum())
    ct_obs   = adata_train.obs["celltype"].iloc[i]
    cid      = train_celltype_labels[i]
    mapped   = id2type[int(cid)]
    ok = "✅" if str(ct_obs) == str(mapped) else "❌"
    print(f"[{i}] n_nonzero={n_nonzero}, obs.celltype={ct_obs!r}, "
          f"label={cid}, id2type={mapped!r}  {ok}")
print("=" * 70 + "\n")
#-------------------------

# ---- Vlidation Data Set ----
all_counts_valid = (
    adata_valid.layers[input_layer_key].toarray()
    if issparse(adata_valid.layers[input_layer_key])
    else adata_valid.layers[input_layer_key]
)
valid_celltype_labels = np.array(adata_valid.obs["celltype_id"].tolist())
valid_batch_labels    = np.array(adata_valid.obs["batch_id"].tolist())

# num_batch_types
num_batch_types = len(set(train_batch_labels) | set(valid_batch_labels))

if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]


vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


tokenized_train = tokenize_and_pad_batch(
    all_counts_train,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)
#-----------------------------------
# ============================================================
# MAPPING CHECK 3: tokenized_train rows <-> train_celltype_labels
# ============================================================
print("\n" + "=" * 70)
print("MAP CHECK 3: tokenized_train vs train_celltype_labels")
print("=" * 70)
print(f"tokenized_train['genes'].shape = {tokenized_train['genes'].shape}")
print(f"train_celltype_labels.shape    = {train_celltype_labels.shape}")
assert tokenized_train["genes"].shape[0] == len(train_celltype_labels), \
    f"ROW COUNT MISMATCH: {tokenized_train['genes'].shape[0]} vs {len(train_celltype_labels)}"

# For 5 sampled cells, verify n_tokens in tokenized matches n_nonzero in X_binned
pad_id = vocab["<pad>"]
for i in indices[:5]:
    n_tokens = int((tokenized_train["genes"][i] != pad_id).sum())
    X_row    = all_counts_train[i]
    n_nonzero = int((X_row > 0).sum())
    # n_tokens should be n_nonzero + 1 (for <cls>) if not truncated
    truncated = "TRUNCATED" if n_tokens >= max_seq_len else ""
    print(f"[{i}] n_tokens={n_tokens}, n_nonzero={n_nonzero}, "
          f"celltype={adata_train.obs['celltype'].iloc[i]!r}  {truncated}")
print("=" * 70 + "\n")
#-----------------------------------
tokenized_valid = tokenize_and_pad_batch(
    all_counts_valid,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)



def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = int(os.environ.get("NUM_WORKERS", 0))
        #num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

# ## Step 3: Load the pre-trained scGPT model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        #for k, v in pretrained_dict.items():
            #
            #logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

#pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

# Freeze all pre-decoder weights
for name, para in model.named_parameters():
    #print("-"*20)
    #print(f"name: {name}")
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
    # if config.freeze and "encoder" in name:
        #print(f"freezing weights for: {name}")
        para.requires_grad = False


model.to(device)


if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=num_batch_types,
    ).to(device)

##############################################################################################################
# ==============================================================================
# Section 3: Loss Function Factory & Hierarchical Loss
# ==============================================================================

def hierarchical_ce_loss(logits, labels, dist_mat, temperature=0.5):
    """
    logits: (batch, n_classes)
    labels: (batch,)
    dist_mat: (n_classes, n_classes) distance matrix
    temperature: softening coefficient, smaller encourages predicting closer categories
    """
    # Temperature scaling
    logits = logits / temperature
    # Compute log_softmax
    log_probs = torch.log_softmax(logits, dim=-1)  # (batch, n_classes)
    
    batch_size = logits.size(0)
    loss = 0.0
    for i in range(batch_size):
        true_label = labels[i]
        dist_vec = dist_mat[true_label]          # (n_classes,)
        weights = torch.exp(-dist_vec)            # (n_classes,)
        weights = weights / weights.sum()
        # Weighted negative log-likelihood
        loss_i = - (weights * log_probs[i]).sum()
        loss += loss_i
    return loss / batch_size

#-----------------------------------------------------------
def build_loss(loss_type, dist_mat=None, temperature=0.5, label_smoothing=0.1):
    loss_type = loss_type.lower()
    if loss_type == 'standard_ce':
        ce = nn.CrossEntropyLoss()
        return lambda logits, labels: ce(logits, labels)  # temperature=1.0
    elif loss_type == 'ce_temp':
        ce = nn.CrossEntropyLoss()
        return lambda logits, labels: ce(logits / temperature, labels)
    elif loss_type == 'ce_uniform_smooth':
        ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return lambda logits, labels: ce(logits / temperature, labels)
    elif loss_type == 'ce_hierarchy':
        if dist_mat is None:
            raise ValueError("ce_hierarchy need dist_mat")
        if not isinstance(dist_mat, torch.Tensor):
            dist_mat = torch.as_tensor(dist_mat, dtype=torch.float32)
        return lambda logits, labels: hierarchical_ce_loss(logits, labels, dist_mat.to(logits.device), temperature=temperature)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Choose from 'standard_ce', 'ce_temp', 'ce_uniform_smooth', 'ce_hierarchy'.")

################################################################################################################
criterion = masked_mse_loss

#--------------------------------------
# Select Group：'standard_ce', 'ce_temp', 'ce_uniform_smooth', 'ce_hierarchy'
LOSS_TYPE = 'ce_hierarchy'
TEMPERATURE = 0.5
LABEL_SMOOTHING = 0.1
criterion_cls = build_loss(
    LOSS_TYPE,
    dist_mat=dist_mat,          # only for 'ce_hierarchy' 
    temperature=TEMPERATURE,
    label_smoothing=LABEL_SMOOTHING,
)
#--------------------------------------
###############################################################################################################
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config.schedule_ratio
)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=config.schedule_ratio
    )
    
if ADV:
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, schedule_interval, gamma=config.schedule_ratio
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D, schedule_interval, gamma=config.schedule_ratio
    )

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                #generative_training=False
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}
            if MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
            if MVC:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
            if MVC and explicit_zero_prob:
                loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_mvc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
            if ECS:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if DAB:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                #generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_mse += loss_mse.item() if MLM else 0.0
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if CCE else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_ecs += loss_ecs.item() if ECS else 0.0
        total_dab += loss_dab.item() if DAB else 0.0
        total_adv_E += loss_adv_E.item() if ADV else 0.0
        total_adv_D += loss_adv_D.item() if ADV else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
        )
        total_error += error_rate
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if CCE else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0
            cur_adv_E = total_adv_E / log_interval if ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if ADV else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval
                if MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"err {cur_error:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                + (f"dab {cur_dab:5.2f} |" if DAB else "")
                + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                    if MVC and explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num


# ## Step 4: Finetune scGPT with task-specific objectives

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=True, 
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_err = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    scheduler.step()
    if DAB_separate_optim:
        scheduler_dab.step()
    if ADV:
        scheduler_D.step()
        scheduler_E.step()

 # inference
def test(model: nn.Module, adata: AnnData) -> Tuple[np.ndarray, np.ndarray, dict]:
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers = int(os.environ.get("NUM_WORKERS", 0)),
        #num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
    )

    # compute accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results

###############################################  Benchmark Test #############################
# ---------- 1. Build tree and node->path mapping from level_sequences ----------
def build_tree_from_level_sequences(level_sequences):
    """
    level_sequences: list of str, each string like "leukocyte->lymphocyte->T cell"
    Returns: label_to_path (dict), root (Node)
    """
    root = Node("ROOT")
    label_to_path = {}
    
    for seq in level_sequences:
        if pd.isna(seq):
            continue
        nodes = [node.strip() for node in seq.split("->")]
        current = root
        for node_name in nodes:
            child = find(current, lambda n: n.name == node_name)
            if child is None:
                child = Node(node_name, parent=current)
            current = child
    # Recursively record the complete path from ROOT for each node
    def traverse(node, current_path):
        new_path = current_path + [node.name]
        label_to_path[node.name] = new_path
        for child in node.children:
            traverse(child, new_path)
    traverse(root, [])
    return label_to_path, root
# ==============================================================================
# Section 4: Hierarchical Evaluation Metrics (Benchmark)
# ==============================================================================
# ---------- 2. Compute depth accuracy ----------
def hierarchical_prf(true_path, pred_path):
    """hierarchical precision / recall / F1 for one sample."""
    true_set = set(true_path)
    pred_set = set(pred_path)
    inter = true_set & pred_set
    hP = len(inter) / len(pred_set) if pred_set else 0.0
    hR = len(inter) / len(true_set) if true_set else 0.0
    if hP + hR == 0:
        return 0.0, 0.0, 0.0
    hF1 = 2 * hP * hR / (hP + hR)
    return hP, hR, hF1


def lca_distance(true_path, pred_path):
    """Shortest path length between true and predicted nodes via LCA."""
    common_depth = -1
    min_len = min(len(true_path), len(pred_path))
    for k in range(min_len):
        if true_path[k] == pred_path[k]:
            common_depth = k
        else:
            break
    depth_t = len(true_path) - 1
    depth_p = len(pred_path) - 1
    return (depth_t - common_depth) + (depth_p - common_depth)

def depth_accuracy_from_adata(adata,
                              global_label_to_path, 
                              pred_col='predictions',
                              true_col='cell_type',
                              seq_col='level_sequence'):
    """
    Returns a DataFrame with columns:
      depth, accuracy, n, hP, hR, hF1, lca_dist
    """
    # 1. Build tree from level_sequence
    unique_seqs = adata.obs[seq_col].dropna().unique()
    #label_to_path, _ = build_tree_from_level_sequences(unique_seqs)
    label_to_path = global_label_to_path
    df = adata.obs[[pred_col, true_col, seq_col]].dropna().copy()

    max_depth = max(len(path) for path in label_to_path.values()) - 1
    if max_depth <= 0:
        return pd.DataFrame(
            columns=['depth', 'accuracy', 'n', 'hP', 'hR', 'hF1', 'lca_dist']
        )

    # 2. Pre-compute per-sample true_path, pred_path
    sample_data = []
    for _, row in df.iterrows():
        true_seq = row[seq_col]
        pred_label = row[pred_col]
        true_path = ["ROOT"] + [n.strip() for n in true_seq.split("->")]
        pred_path = label_to_path.get(pred_label)
        sample_data.append((true_path, pred_path))

    # 3. Per-depth metrics
    depth_acc = []
    for d in range(1, max_depth + 1):
        matches = []
        hP_list, hR_list, hF1_list, lca_list = [], [], [], []

        for true_path, pred_path in sample_data:
            # 3a. depth accuracy: only cells whose true path reaches depth d
            if len(true_path) <= d:
                continue

            if pred_path is None or len(pred_path) <= d:
                # 
                matches.append(False)
                # 
                hP_list.append(0.0)
                hR_list.append(0.0)
                hF1_list.append(0.0)
                lca_list.append(float(2.0 * max_depth))  # 
                continue

            # 
            matches.append(true_path[d] == pred_path[d])

            # 
            true_sub = true_path[:d+1]
            pred_sub = pred_path[:d+1]

            hP, hR, hF1 = hierarchical_prf(true_sub, pred_sub)
            hP_list.append(hP)
            hR_list.append(hR)
            hF1_list.append(hF1)
            lca_list.append(lca_distance(true_sub, pred_sub))

        n_at_depth = len(matches)
        acc = sum(matches) / n_at_depth if n_at_depth > 0 else 0.0
        hP_mean = float(np.mean(hP_list)) if hP_list else 0.0
        hR_mean = float(np.mean(hR_list)) if hR_list else 0.0
        hF1_mean = float(np.mean(hF1_list)) if hF1_list else 0.0
        lca_mean = float(np.mean(lca_list)) if lca_list else 0.0

        depth_acc.append({
            'depth': d,
            'accuracy': acc,
            'n': n_at_depth,
            'hP': hP_mean,
            'hR': hR_mean,
            'hF1': hF1_mean,
            'lca_dist': lca_mean,
        })

    return pd.DataFrame(depth_acc)
#--------------------------------------------------
def calculate_global_metrics(pred_df, global_label_to_path):
    """
    Calculate depth-wise and global hierarchical evaluation metrics.
    
    Args:
        pred_df (pd.DataFrame): DataFrame containing 'predictions' and 'level_sequence' columns.
        global_label_to_path (dict): Mapping from a cell type label to its full hierarchical path (from "ROOT").
        
    Returns:
        tuple: (depth_df, global_metrics)
            - depth_df (pd.DataFrame): Metrics per depth (depth, accuracy, n_cells, n_classes, n_cells_eq_d, classes_at_depth).
            - global_metrics (dict): Global metrics (n_total_cells, max_depth, hP, hR, hF1, lca_dist).
    """
    max_depth = max(len(path) for path in global_label_to_path.values()) - 1
    df = pred_df.dropna(subset=['predictions', 'level_sequence']).copy()

    # Construct true_path and pred_path for each cell
    sample_data = []
    for _, row in df.iterrows():
        true_seq = row['level_sequence']
        pred_label = row['predictions']
        true_path = ["ROOT"] + [n.strip() for n in str(true_seq).split("->")]
        pred_path = global_label_to_path.get(pred_label)
        sample_data.append((true_path, pred_path))

    # ==========================================================
    # Global Metrics: Computed on full paths for each cell.
    # - No depth truncation.
    # - If pred_path is None, apply maximum penalty (2.0 * max_depth).
    # ==========================================================
    hP_all, hR_all, hF1_all, lca_all = [], [], [], []
    for true_path, pred_path in sample_data:
        if pred_path is None:
            hP_all.append(0.0)
            hR_all.append(0.0)
            hF1_all.append(0.0)
            lca_all.append(2.0 * max_depth)      
            continue
        
        hP, hR, hF1 = hierarchical_prf(true_path, pred_path)   # Use full paths
        hP_all.append(hP)
        hR_all.append(hR)
        hF1_all.append(hF1)
        lca_all.append(lca_distance(true_path, pred_path))     # Use full paths

    global_metrics = {
        'n_total_cells': len(sample_data),
        'max_depth':     max_depth,
        'hP':            float(np.mean(hP_all)) if hP_all else float('nan'),
        'hR':            float(np.mean(hR_all)) if hR_all else float('nan'),
        'hF1':           float(np.mean(hF1_all)) if hF1_all else float('nan'),
        'lca_dist':      float(np.mean(lca_all)) if lca_all else float('nan'),
    }

    # ==========================================================
    # Depth-wise Metrics: Compute accuracy, cumulative cell count, and class count per depth.
    # - If n=0, return NaN instead of 0.0 to prevent skewing overall averages.
    # - Track 'n_classes' and 'classes_at_depth' for detailed distribution analysis.
    # ==========================================================
    depth_acc = []
    for d in range(1, max_depth + 1):
        matches = []
        classes = set()
        n_eq_d = 0
        for true_path, pred_path in sample_data:
            if len(true_path) <= d:
                continue                           # Keep only cells where true_depth >= d
            classes.add(true_path[-1])             # Numerator for n_classes
            if len(true_path) - 1 == d:
                n_eq_d += 1                        # Cells where true depth exactly equals d
            
            # Check if prediction exists at this depth and matches the ground truth
            ok = (pred_path is not None
                  and len(pred_path) > d
                  and true_path[d] == pred_path[d])
            matches.append(ok)

        n = len(matches)
        depth_acc.append({
            'depth':            d,
            'accuracy':         sum(matches) / n if n > 0 else float('nan'),
            'n_cells':          n,                                            # Cumulative count >= d
            'n_classes':        len(classes),                                 
            'n_cells_eq_d':     n_eq_d,
            'classes_at_depth': ';'.join(sorted(classes)) if classes else '',
        })

    return pd.DataFrame(depth_acc), global_metrics

# ==============================================================================
# Section 5: Main Pipeline Execution (Inference with fine-tuned scGPT model)
# ==============================================================================
##############################################################################################################################################
# In the cell-type annotation task, the fine-tuned scGPT predicts cell-type labels for query set as inference. 
# The model performance is evaluated on standard classificaton metrics. Here we visualize the predicted labels over the scGPT cell embeddings, 
# and present the confusion matrix for detailed classification performance on the cell-group level.
predictions, labels, results = test(best_model, adata_test)
adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]
print("Building global tree from all splits...")
# Map predictions back to the *raw* complete set to ensure all cell types and 
# hierarchical paths are preserved for evaluation.
global_unique_seqs = adata.obs['level_sequence'].dropna().unique() 
global_label_to_path, _ = build_tree_from_level_sequences(global_unique_seqs)
pred_df = adata_test.obs[['cell_type', 'level_sequence']].copy()
pred_df.insert(0, 'cell_name', adata_test.obs_names)
pred_df['predictions'] = [id2type[p] for p in predictions]
pred_df['labels'] = [id2type[p] for p in labels]
pred_df = pred_df[['cell_name', 'predictions', 'labels', 'cell_type', 'level_sequence']]
pred_df.to_csv(save_dir / "predictions.csv", index=False)

# # Compute depth-wise hierarchical metrics using the complete raw  data including train, validation and test data
depth_acc_df = depth_accuracy_from_adata(adata_test_raw, global_label_to_path)
print(depth_acc_df)
depth_acc_df.to_csv(save_dir / "depth_accuracy.csv", index=False)

# # Compute global LCA 
Ndepth_df, global_metrics = calculate_global_metrics(pred_df, global_label_to_path)
Ndepth_df.to_csv(save_dir / "Ndepth_counts_statistics.csv", index=False)
pd.DataFrame([global_metrics]).to_csv(save_dir / "global_hier_metrics.csv", index=False)


# save the model into the save_dir
torch.save(best_model.state_dict(), save_dir / "model.pt")