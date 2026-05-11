"""
Final Check and Correction Module for Processed .h5ad Files in a Single-cell RNA-seq Preprocessing Pipeline
"""
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# ------------------------------ Configuration ---------------------------------
CONFIG = {
    "data_dirs": {
        "raw": "SCFM/HCA/",                # original h5ad files
        "preprocessing": "SCFM/HCA/preprocessing/",
        "var_h5ad": "SCFM/var_h5ad/",
        "final": "SCFM/var_h5ad_HCA/new/"
    },
    "files": {
        "var_match": "SCFM/var_match/ALL_var_match.csv",
        "cl_match_level": "SCFM/statistics/CL_match_level.txt",
        "double_cells": None,   # optional: path to a CSV with double cell info, e.g., columns ['cell_dataset', 'is_duplicate']
    },
    "qc": {
        "min_genes": 200,
        "max_pct_mt": 20
    },
    "hvg_n_top": [2048, 4096],
    "celltype_blacklist": ["unknown"],
    "column_mapping": {
        # cell type
        "Celltype": "cell_type",
        "celltype": "cell_type",
        "cell_type": "cell_type",
        # mitochondrial percent
        "X.Mitochondrial": "pct_counts_mt",
        "percent.mt": "pct_counts_mt",
        "percent.mito": "pct_counts_mt",
        # n_genes
        "CountGenes": "n_genes_by_counts",
        "nFeature_RNA": "n_genes_by_counts",
        "n_genes_by_counts": "n_genes_by_counts",
        # UMIs
        "CountUMIs": "total_counts",
        "nCount_RNA": "total_counts",
        "total_counts": "total_counts",
        # platform
        "assay": "platform",
    }
}

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------ Helper functions ------------------------------
def load_adata(path: Path) -> sc.AnnData:
    """Load h5ad and ensure .raw exists."""
    adata = sc.read_h5ad(path)
    if adata.raw is None:
        adata.raw = adata.copy()
        logger.debug(f"Created .raw for {path.name}")
    return adata

def standardize_obs(adata: sc.AnnData, mapping: Dict[str, str], compute_qc_if_missing: bool = True) -> sc.AnnData:
    """
    Rename obs columns according to mapping. If QC metrics missing, compute them.
    Assumes raw counts are in adata.layers['raw_counts'] or adata.raw.X.
    """
    for old, new in mapping.items():
        if old in adata.obs.columns and new not in adata.obs.columns:
            adata.obs[new] = adata.obs[old]
    
    if compute_qc_if_missing and "n_genes_by_counts" not in adata.obs.columns:
        # Temporarily use raw counts for QC calculation
        if "raw_counts" in adata.layers:
            original_X = adata.X
            adata.X = adata.layers["raw_counts"]
        elif adata.raw is not None:
            original_X = adata.X
            adata.X = adata.raw.X
        else:
            logger.warning("No raw counts found, skipping QC calculation")
            return adata
        
        adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        # Restore original X (e.g., log-normalized data)
        adata.X = original_X
    return adata

def filter_celltype(adata: sc.AnnData, blacklist: List[str]) -> sc.AnnData:
    """Remove cells whose cell_type is in blacklist."""
    if "cell_type" not in adata.obs:
        logger.warning("Column 'cell_type' not found, skipping filtering")
        return adata
    mask = ~adata.obs["cell_type"].astype(str).str.strip().isin(blacklist)
    return adata[mask].copy()

def match_ensg(adata: sc.AnnData, var_match_df: pd.DataFrame) -> sc.AnnData:
    """
    Map gene names to ENSG IDs using a lookup table.
    Discards genes not found in the table and handles duplicate ENSG IDs.
    """
    gene_to_ensg = dict(zip(var_match_df["gene_name"].astype(str), var_match_df["ensg"]))
    # Keep genes present in mapping
    keep = adata.var_names.isin(gene_to_ensg.keys())
    adata = adata[:, keep].copy()
    adata.var["ensg"] = adata.var_names.map(gene_to_ensg)
    # Remove rows where ensg is NaN
    adata = adata[:, ~adata.var["ensg"].isna()].copy()
    # Make var_names unique using ENSG IDs
    if not adata.var["ensg"].is_unique:
        adata.var_names = adata.var["ensg"].astype(str)
        adata.var_names_make_unique()
    else:
        adata.var_names = adata.var["ensg"].astype(str)
    return adata

def add_hvg(adata: sc.AnnData, n_top_list: List[int]) -> sc.AnnData:
    """
    Compute highly variable genes (Seurat flavor) for each n_top value.
    Assumes adata.X contains log-normalized counts; uses raw_counts for normalization if available.
    """
    # Ensure we work from raw counts for normalization
    if "raw_counts" in adata.layers:
        original_X = adata.X
        adata.X = adata.layers["raw_counts"]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # Store normalized data back to X (original will be restored later? We'll keep normalized)
        # But to avoid side effects, we can keep it as is; many pipelines want X = log1p normalized.
        # We'll not restore original_X here.
    
    for n in n_top_list:
        sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor="seurat", inplace=True)
        adata.var.rename(columns={"highly_variable": f"HVG_{n}"}, inplace=True)
    return adata

def is_integer_matrix(matrix: sp.spmatrix or np.ndarray, sample_size: int = 1000) -> bool:
    """Check if a sparse/dense matrix contains only integer values (sampling non-zero entries)."""
    if sp.issparse(matrix):
        data = matrix.data
    else:
        data = matrix.flatten()
    non_zero = data[data != 0]
    if len(non_zero) == 0:
        return True
    sample = non_zero[:sample_size]
    return np.all(np.equal(np.round(sample), sample))

def fix_raw_counts(adata: sc.AnnData) -> sc.AnnData:
    """
    Repair adata.layers['raw_counts'] using adata.raw.X.
    Aligns gene order and ensures integer type.
    """
    if adata.raw is None:
        raise ValueError("adata.raw is required for fixing raw_counts")
    if "raw_counts" not in adata.layers:
        logger.warning("No raw_counts layer found, creating from raw.X")
        adata.layers["raw_counts"] = adata.raw.X.astype(int)
        return adata
    raw_X = adata.raw.X
    raw_genes = adata.raw.var_names
    # Current var names (prefer original_index if present, otherwise var_names)
    cur_genes = adata.var.get("original_index", pd.Series(adata.var_names, index=adata.var_names))
    cur_genes = cur_genes.astype(str)
    # Find common genes
    common = cur_genes[cur_genes.isin(raw_genes)]
    if len(common) == 0:
        raise RuntimeError("No common genes between current var and raw.var_names")
    # Indices in raw and current matrices
    raw_idx = raw_genes.get_indexer(common.values)
    cur_idx = cur_genes.index.get_indexer(common.index)
    # Subset raw matrix
    if sp.issparse(raw_X):
        sub = raw_X[:, raw_idx].tocoo()
    else:
        sub = raw_X[:, raw_idx]
    
    # Build new matrix with correct column ordering
    new_counts = sp.coo_matrix((sub.data, (sub.row, cur_idx[sub.col])),
                                shape=adata.shape).tocsr().astype(int)
    adata.layers["raw_counts"] = new_counts
    logger.info("Fixed raw_counts layer using adata.raw.X")
    return adata

def remove_duplicate_cells(adata: sc.AnnData, double_df: Optional[pd.DataFrame] = None) -> sc.AnnData:
    """
    Remove cells that are marked as duplicates based on 'cell_dataset' column.
    double_df should have columns: 'cell_dataset' and 'is_duplicate' (boolean).
    If double_df not provided, only remove exact duplicated obs_names.
    """
    # Remove exact duplicated obs_names first
    if adata.obs_names.duplicated().any():
        adata = adata[~adata.obs_names.duplicated()].copy()
        logger.info("Removed cells with duplicated obs_names")
    if double_df is not None:
        if "cell_dataset" not in adata.obs:
            logger.warning("Column 'cell_dataset' not found, cannot remove based on double_df")
            return adata
        dup_cells = double_df.loc[double_df["is_duplicate"], "cell_dataset"].values
        mask = ~adata.obs["cell_dataset"].isin(dup_cells)
        adata = adata[mask].copy()
        logger.info(f"Removed {len(dup_cells)} cells marked as duplicates")
    return adata

def correct_level_sequence(adata: sc.AnnData, wrong_str: str, correct_str: str) -> sc.AnnData:
    """Replace an incorrect level_sequence string with the correct one."""
    if "level_sequence" not in adata.obs:
        logger.warning("Column 'level_sequence' not found, skipping correction")
        return adata
    wrong_mask = adata.obs["level_sequence"].astype(str).str.strip() == wrong_str
    if wrong_mask.any():
        adata.obs.loc[wrong_mask, "level_sequence"] = correct_str
        logger.info(f"Corrected {wrong_mask.sum()} rows of level_sequence")
    return adata

def check_and_fix_integrity(adata: sc.AnnData) -> sc.AnnData:
    """Check if raw_counts are integer; if not, repair using raw.X."""
    if "raw_counts" not in adata.layers:
        logger.warning("No raw_counts layer, cannot check integrity")
        return adata
    if not is_integer_matrix(adata.layers["raw_counts"]):
        logger.info("raw_counts not integer, attempting fix")
        adata = fix_raw_counts(adata)
    else:
        logger.info("raw_counts already integer")
    return adata

# ------------------------------ Main processing pipeline ----------------------
def process_file(
    file_path: Path,
    config: Dict,
    var_match_df: pd.DataFrame,
    double_df: Optional[pd.DataFrame] = None,
    level_correction: Optional[Tuple[str, str]] = None,  # (wrong_str, correct_str)
    steps: List[str] = None
) -> sc.AnnData:
    """
    Apply a sequence of preprocessing steps to a single h5ad file.
    Steps can include: 'standardize', 'filter_celltype', 'match_ensg', 'add_hvg',
    'remove_duplicates', 'correct_level', 'fix_integrity'.
    """
    if steps is None:
        steps = ["standardize", "filter_celltype", "match_ensg", "add_hvg", "fix_integrity"]
    
    adata = load_adata(file_path)
    logger.info(f"Processing {file_path.name} | cells: {adata.n_obs}")
    
    # Store original gene names for possible later use
    adata.var["original_index"] = adata.var_names
    
    if "standardize" in steps:
        adata = standardize_obs(adata, config["column_mapping"])
    if "filter_celltype" in steps:
        adata = filter_celltype(adata, config["celltype_blacklist"])
    if "match_ensg" in steps:
        adata = match_ensg(adata, var_match_df)
    if "add_hvg" in steps:
        adata = add_hvg(adata, config["hvg_n_top"])
    if "remove_duplicates" in steps and double_df is not None:
        adata = remove_duplicate_cells(adata, double_df)
    if "correct_level" in steps and level_correction is not None:
        wrong, correct = level_correction
        adata = correct_level_sequence(adata, wrong, correct)
    if "fix_integrity" in steps:
        adata = check_and_fix_integrity(adata)
    
    logger.info(f"After processing: cells={adata.n_obs}, genes={adata.n_vars}")
    return adata

def main():
    # Load global lookup tables
    var_match_df = pd.read_csv(CONFIG["files"]["var_match"])
    
    # Optional: load duplicate cell info and level correction parameters
    double_df = None
    level_correction = None
    # Example if you have these files
    # double_df = pd.read_csv("path/to/double_cells.csv")
    # level_correction = ("old_string", "new_string")
    
    # Define input and output directories
    raw_dir = Path(CONFIG["data_dirs"]["raw"])
    out_dir = Path(CONFIG["data_dirs"]["final"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather all h5ad files
    h5ad_files = list(raw_dir.glob("*.h5ad"))
    logger.info(f"Found {len(h5ad_files)} files to process")
    
    for f in tqdm(h5ad_files, desc="Processing files"):
        try:
            # Optionally skip if output already exists
            out_path = out_dir / f.name
            if out_path.exists():
                logger.info(f"Skipping {f.name} (already exists)")
                continue
            
            adata = process_file(
                file_path=f,
                config=CONFIG,
                var_match_df=var_match_df,
                double_df=double_df,
                level_correction=level_correction,
                steps=["standardize", "filter_celltype", "match_ensg", "add_hvg", "fix_integrity"]
            )
            adata.write(out_path)
            logger.info(f"Saved {out_path}")
        except Exception as e:
            logger.error(f"Failed to process {f.name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()