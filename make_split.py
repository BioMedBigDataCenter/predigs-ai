#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_splits.py
--------------
Split a single large h5ad into raw.train.h5ad / raw.valid.h5ad / raw.test.h5ad

Steps:
  [1/6] Load + var_names_make_unique
  [2/6] HVG filtering (skip with --skip_hvg)
  [3a/6] level_sequence statistics (depth distribution of drop vs keep, skip with --skip_level_stats)
  [3/6] Drop cell types with < min_per_type cells
  [4/6] Randomly subsample cell types with > max_per_type cells
  [5/6] Unify celltype encoding + stratified split 45/5/50
  [6/6] sanitize categoricals + write three h5ad files
"""

import argparse
import gc
import sys
from collections import Counter
from pathlib import Path
import scanpy as sc
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def allocate_counts(n, fracs, min_valid=1):
    """Split n into len(fracs) integers using the largest remainder method.

    The sum is always n, and each part is >= 1.
    """
    k = len(fracs)
    if n < k:
        die(f"n={n} is too small to split into {k} parts")
    min_valid = min(min_valid, n - (k - 1))

    quotas = np.asarray(fracs, dtype=float) * n
    base = np.floor(quotas).astype(int)
    base[np.argsort(-(quotas - base), kind="stable")[: n - base.sum()]] += 1

    if base[1] < min_valid:
        need = min_valid - base[1]
        take = min(need, base[0] - 1)
        base[0] -= take
        base[1] += take
        need -= take
        if need > 0:
            take = min(need, base[2] - 1)
            base[2] -= take
            base[1] += take

    for i in range(k):
        if base[i] < 1:
            base[int(np.argmax(base))] -= 1
            base[i] = 1

    assert base.sum() == n and (base >= 1).all()
    return base


def bool_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).values.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool).values
    return (series.astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "y"]).values)


def sanitize_categoricals(adata):
    for attr in ("obs", "var"):
        df = getattr(adata, attr)
        for col in df.columns:
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                continue
            cats = df[col].cat.categories
            if all(isinstance(c, str) for c in cats):
                continue
            try:
                df[col] = df[col].cat.rename_categories([str(c) for c in cats])
            except ValueError:
                df[col] = df[col].astype(str).astype("category")
    return adata


def fmt_pct(x):
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- #
# level_sequence statistics
# --------------------------------------------------------------------------- #
def _seq_depth(seq):
    """'ROOT->A->B' -> 3; NaN/empty -> None."""
    if pd.isna(seq):
        return None
    s = str(seq).strip()
    if not s:
        return None
    return len([x for x in s.split("->") if x.strip()])


def _print_depth_dist(tag, depths, indent="      "):
    if not depths:
        print(f"{indent}[{tag}] (empty)")
        return
    s = pd.Series(depths)
    c = Counter(depths)
    total = len(depths)
    print(f"{indent}[{tag}] n={total}  min={s.min()}  max={s.max()}  "
          f"median={int(s.median())}  mean={s.mean():.2f}")
    for d in sorted(c):
        bar = "#" * max(1, int(30 * c[d] / total))
        print(f"{indent}    depth={d:2d}  n={c[d]:4d}  "
              f"({c[d] / total * 100:5.1f}%)  {bar}")


def stats_level_sequence(obs, celltype_col, level_col, min_per_type):
    """Compute level_sequence depth distribution for drop vs keep on raw obs.

    Returns a DataFrame with one row per cell type:
    celltype / n_cells / depth / status / seq, etc.
    """
    if level_col not in obs.columns:
        print(f"      [WARN] obs has no '{level_col}'; skipping level_sequence stats")
        return None

    rows = []
    for t, sub in obs.groupby(celltype_col, observed=True):
        seqs = sub[level_col].dropna().astype(str).unique()
        seq = seqs[0] if len(seqs) > 0 else None
        rows.append({
            "celltype": str(t),
            "n_cells": len(sub),
            "level_sequence": seq,
            "depth": _seq_depth(seq),
            "n_unique_seq": len(seqs),
        })
    df = pd.DataFrame(rows)
    df["status"] = np.where(df["n_cells"] >= min_per_type, "keep", "drop")

    # Consistency / NaN warnings
    bad = df.loc[df["n_unique_seq"] > 1, "celltype"].tolist()
    if bad:
        print(f"      [WARN] {len(bad)} types have >1 distinct level_sequence:")
        print(f"             {bad[:5]}{'...' if len(bad) > 5 else ''}")

    nan_keep = df.loc[(df.status == "keep") & df.depth.isna(), "celltype"].tolist()
    if nan_keep:
        print(f"      [WARN] kept types with NaN level_sequence "
              f"({len(nan_keep)}): {nan_keep}")

    # Print
    d_drop = df.loc[df.status == "drop", "depth"].dropna().tolist()
    d_keep = df.loc[df.status == "keep", "depth"].dropna().tolist()

    print(f"      celltype: total={len(df)}  "
          f"drop(<{min_per_type})={int((df.status=='drop').sum())}  "
          f"keep(>={min_per_type})={int((df.status=='keep').sum())}")
    _print_depth_dist("DROP", d_drop)
    _print_depth_dist("KEEP", d_keep)

    if d_drop and d_keep:
        sd, sk = pd.Series(d_drop), pd.Series(d_keep)
        print(f"      comparison:  drop mean={sd.mean():.2f} median={int(sd.median())} "
              f"| keep mean={sk.mean():.2f} median={int(sk.median())}")

    return df


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def subsample_and_split(
    input_path,
    out_dir="./",
    celltype_col="cell_type",
    level_col="level_sequence",
    hvg_col="HVG_4096",
    skip_hvg=False,
    skip_level_stats=False,
    level_stats_csv="",
    min_per_type=30,
    max_per_type=200,
    train_frac=0.45,
    valid_frac=0.05,
    test_frac=0.50,
    min_valid_per_type=1,
    encode_celltype_id=True,
    force=False,
    seed=42,
):
    # ---------- Parameter validation ----------
    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-8:
        die("The fractions must sum to 1")
    if not (0 < min_per_type <= max_per_type):
        die("Require 0 < min_per_type <= max_per_type")

    in_path = Path(input_path)
    if not in_path.exists():
        die(f"Input file does not exist: {in_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [out_dir / f for f in
               ("raw.train.h5ad", "raw.valid.h5ad", "raw.test.h5ad")]
    if not force:
        exists = [str(p) for p in targets if p.exists()]
        if exists:
            die(f"Output already exists; add --force to overwrite: {exists}")

    rng = np.random.default_rng(seed)

    # ---------- [1/6] Load ----------
    print(f"[1/6] load {in_path}")
    adata = sc.read_h5ad(in_path)
    print(f"      {adata.n_obs} cells x {adata.n_vars} genes")
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # ---------- [2/6] HVG filtering ----------
    if skip_hvg:
        print(f"[2/6] skip HVG filtering (keep all {adata.n_vars} genes)")
    else:
        if hvg_col not in adata.var.columns:
            die(f"Column '{hvg_col}' not found in var")
        n_before = adata.n_vars
        mask = bool_mask(adata.var[hvg_col])
        if mask.sum() == 0:
            die(f"var['{hvg_col}'] has no True values")
        adata = adata[:, np.where(mask)[0]].copy()
        gc.collect()
        print(f"[2/6] keep var['{hvg_col}']==True: {n_before} -> {adata.n_vars}")

    # ---------- Validate celltype column ----------
    if celltype_col not in adata.obs.columns:
        die(f"Column '{celltype_col}' not found in obs; existing columns: {list(adata.obs.columns)}")
    if adata.obs[celltype_col].isna().any():
        die(f"'{celltype_col}' contains NaN")

    # ---------- [3a/6] level_sequence stats (before dropping) ----------
    if not skip_level_stats:
        print(f"[3a/6] level_sequence stats "
              f"(celltype_col='{celltype_col}', level_col='{level_col}')")
        level_df = stats_level_sequence(
            adata.obs[[celltype_col] + ([level_col] if level_col in adata.obs.columns else [])],
            celltype_col, level_col, min_per_type)
        if level_df is not None and level_stats_csv:
            csv_path = Path(level_stats_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            level_df.to_csv(csv_path, index=False)
            print(f"      saved -> {csv_path.resolve()}")

    # ---------- [3/6] Drop cell types with < min_per_type cells ----------
    ct = adata.obs[celltype_col].astype(str).values
    counts0 = pd.Series(ct).value_counts()
    print(f"      {len(counts0)} celltypes | min={counts0.min()} "
          f"max={counts0.max()} median={int(counts0.median())}")

    print(f"[3/6] drop celltypes with < {min_per_type} cells")
    keep_types = counts0.index[counts0 >= min_per_type]
    if len(keep_types) == 0:
        die(f"No cell type reaches min_per_type={min_per_type}")
    dropped = counts0.index[counts0 < min_per_type].tolist()
    adata = adata[np.where(np.isin(ct, keep_types))[0]].copy()
    ct = adata.obs[celltype_col].astype(str).values
    print(f"      removed {len(dropped)} types: {dropped}")

    # ---------- [4/6] Subsampling ----------
    print(f"[4/6] subsample celltypes with > {max_per_type} cells")
    keep_pos = []
    for t in np.unique(ct):
        p = np.where(ct == t)[0]
        if len(p) > max_per_type:
            p = rng.choice(p, max_per_type, replace=False)
        keep_pos.append(p)
    adata = adata[np.sort(np.concatenate(keep_pos))].copy()
    gc.collect()
    ct = adata.obs[celltype_col].astype(str).values
    counts1 = pd.Series(ct).value_counts()
    print(f"      {adata.n_obs} cells | {len(counts1)} celltypes | "
          f"min={counts1.min()} max={counts1.max()} "
          f"max/min={counts1.max() / counts1.min():.1f}x")

    # ---------- [5/6] Encoding + split ----------
    if encode_celltype_id:
        print("[5/6] unify celltype category & celltype_id")
        sorted_types = sorted(counts1.index.tolist())
        adata.obs[celltype_col] = pd.Categorical(
            adata.obs[celltype_col].astype(str), categories=sorted_types)
        adata.obs["celltype_id"] = adata.obs[celltype_col].cat.codes.astype(int)
    else:
        print("[5/6] skip celltype_id encoding")

    print(f"      stratified split "
          f"{fmt_pct(train_frac)}/{fmt_pct(valid_frac)}/{fmt_pct(test_frac)}")
    fracs = [train_frac, valid_frac, test_frac]
    tr, va, te = [], [], []
    for t in np.unique(ct):
        idx = np.where(ct == t)[0].copy()
        rng.shuffle(idx)
        n_tr, n_va, n_te = allocate_counts(len(idx), fracs,
                                           min_valid=min_valid_per_type)
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])

    tr = np.sort(np.concatenate(tr))
    va = np.sort(np.concatenate(va))
    te = np.sort(np.concatenate(te))

    assert len(tr) + len(va) + len(te) == adata.n_obs
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(va, te)) == 0

    # ---------- [6/6] Write h5ad ----------
    sanitize_categoricals(adata)
    print(f"[6/6] save to {out_dir.resolve()}")
    for idx_arr, name in [(tr, "raw.train.h5ad"),
                          (va, "raw.valid.h5ad"),
                          (te, "raw.test.h5ad")]:
        adata[idx_arr].copy().write_h5ad(out_dir / name)
        gc.collect()

    n = adata.n_obs
    print("-" * 64)
    print(f"totals: train={len(tr)} ({fmt_pct(len(tr)/n)}) | "
          f"valid={len(va)} ({fmt_pct(len(va)/n)}) | "
          f"test={len(te)} ({fmt_pct(len(te)/n)})")
    print(f"genes={adata.n_vars}  cells={n}")
    print(f"done -> {out_dir.resolve()}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="split h5ad + level_sequence stats")
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default="./")
    p.add_argument("--celltype_col", default="cell_type")
    p.add_argument("--level_col", default="level_sequence",
                   help="Name of the hierarchy sequence column (default: level_sequence)")
    p.add_argument("--hvg_col", default="HVG_4096")
    p.add_argument("--skip_hvg", action="store_true")
    p.add_argument("--skip_level_stats", action="store_true",
                   help="Skip level_sequence depth statistics")
    p.add_argument("--level_stats_csv", default="",
                   help="Export per-celltype level_sequence stats to CSV (optional)")
    p.add_argument("--min_per_type", type=int, default=30)
    p.add_argument("--max_per_type", type=int, default=200)
    p.add_argument("--train_frac", type=float, default=0.45)
    p.add_argument("--valid_frac", type=float, default=0.05)
    p.add_argument("--test_frac",  type=float, default=0.50)
    p.add_argument("--min_valid_per_type", type=int, default=1)
    p.add_argument("--no_encode_celltype_id", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    subsample_and_split(
        input_path=args.input,
        out_dir=args.out_dir,
        celltype_col=args.celltype_col,
        level_col=args.level_col,
        hvg_col=args.hvg_col,
        skip_hvg=args.skip_hvg,
        skip_level_stats=args.skip_level_stats,
        level_stats_csv=args.level_stats_csv,
        min_per_type=args.min_per_type,
        max_per_type=args.max_per_type,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        min_valid_per_type=args.min_valid_per_type,
        encode_celltype_id=not args.no_encode_celltype_id,
        force=args.force,
        seed=args.seed,
    )