library(Seurat)
library(SeuratDisk)
library(DoubletFinder)
library(dplyr)
library(Matrix)
# ===================== 2. Load Seurat object =====================
seurat_obj <- CreateSeuratObject(counts = raw_counts, project = "SCFM", min.cells = 0, min.features = 200)
#load(seurat_obj)
#seurat_obj <- readRDS("Data_path")
# ===================== 3. Quality control=====================
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
# filter cells：
seurat_obj <- subset(seurat_obj,
                     subset = nFeature_RNA > 200 &
                              nCount_RNA > 1000 &
                              percent.mt < 20)
# ===================== 4. Remove doublet (DoubletFinder v2.2.0)=====================
# 7% (per 10,000 cells)
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 2000)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj, npcs = 30)
nExp_poi <- round(0.07 * ncol(seurat_obj))   
sweep.res <- paramSweep_v3(seurat_obj, PCs = 1:30, sct = FALSE)
sweep.stats <- summarizeSweep(sweep.res, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
pK_opt <- as.numeric(as.character(bcmvn$pK[bcmvn$BCmetric == max(bcmvn$BCmetric)]))
# DoubletFinder
seurat_obj <- doubletFinder_v3(seurat_obj,
                               PCs = 1:30,
                               pK = pK_opt,
                               nExp = nExp_poi,
                               reuse.pANN = FALSE)
DF_name <- grep("DF.classifications", colnames(seurat_obj@meta.data), value = TRUE)
seurat_obj <- subset(seurat_obj, cells = rownames(seurat_obj@meta.data[seurat_obj@meta.data[, DF_name] == "Singlet", ]))

# ===================== 5. Further filtering（gene & cell level） =====================
if ("gene_type" %in% colnames(seurat_obj[["RNA"]][[]])) {
  coding_genes <- rownames(seurat_obj)[seurat_obj[["RNA"]][[]]$gene_type == "protein_coding"]
  seurat_obj <- subset(seurat_obj, features = coding_genes)
} else {
  warning("No gene type information")
}
## 5.2 remove deplicate cells
if ("cell_name" %in% colnames(seurat_obj@meta.data)) {
  seurat_obj <- seurat_obj[, !duplicated(seurat_obj$cell_name)]
}
## 5.3 remove ambiguous cell labels
ambiguous_labels <- c("unknown", "none", "monocyte/macrophage")
if ("cell_type" %in% colnames(seurat_obj@meta.data)) {
  seurat_obj <- subset(seurat_obj, subset = !(cell_type %in% ambiguous_labels))
}
## 5.4 remove malignant cell（according to label）
malignant_keywords <- c("malignant cell", "cancer cell", "tumor cell")
if ("cell_type" %in% colnames(seurat_obj@meta.data)) {
  seurat_obj <- subset(seurat_obj, subset = !(cell_type %in% malignant_keywords))
}
# ===================== 6. Gene expression matrix and normalization =====================
if (all(GetAssayData(seurat_obj, layer = "counts") %% 1 == 0)) {
  seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 1e4)
} else {
  seurat_obj[["RNA"]]$counts <- NULL
}
# ===================== 7. Downstream process =====================
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
# HVG
hvgs_2048 <- VariableFeatures(FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2048))
hvgs_4096 <- VariableFeatures(FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 4096))
all_genes <- rownames(seurat_obj)
is_hvg_2048 <- all_genes %in% hvgs_2048
is_hvg_4096 <- all_genes %in% hvgs_4096
seurat_obj[["RNA"]]@meta.features$HVG_2048 <- is_hvg_2048
seurat_obj[["RNA"]]@meta.features$HVG_4096 <- is_hvg_4096
# PCA clustering
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30)
seurat_obj <- FindClusters(seurat_obj, resolution =1)  
seurat_obj <- RunUMAP(seurat_obj, dims = 1:30, verbose = FALSE)
# ===================== 8. Split dataset and save as .h5ad file =====================
if (!"donor_id" %in% colnames(seurat_obj@meta.data)) {
  stop("Without 'sample' column")
}

unique_samples <- unique(seurat_obj$donor_id)
for (samp in unique_samples) {
  subset_obj <- subset(seurat_obj, cells = colnames(seurat_obj)[seurat_obj$donor_id == samp])
  h5seurat_file <- paste0("dataset_", samp, ".h5Seurat")
  h5ad_file <- paste0("dataset_", samp, ".h5ad")
  SaveH5Seurat(subset_obj, filename = h5seurat_file, overwrite = TRUE)
  Convert(h5seurat_file, dest = "h5ad", overwrite = TRUE)
  file.remove(h5seurat_file)
  cat("Save", samp, "to", h5ad_file, "\n")
}
# ===================== END =====================