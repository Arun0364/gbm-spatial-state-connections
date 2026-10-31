suppressPackageStartupMessages({
    library(Seurat)
    library(SeuratObject)
    library(Matrix)
    library(ggplot2)
    library(patchwork)
    library(SingleR)
    library(celldex)
    library(SummarizedExperiment)
})

visium_dir <- "/Users/arun/Desktop/Research/Data/G4_ST2"
outdir <- "/Users/arun/Desktop/Research/Results/CellTypeAssignment/G4_ST2_SingleR"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

find_one <- function(dir, patterns) {
    files <- list.files(dir, full.names = TRUE) # nolint # nolint
    hit <- unlist(lapply(patterns, function(p) grep(p, files, value = TRUE, ignore.case = TRUE))) # nolint # nolint
    if (length(hit) == 0) stop(paste('Could not find:', paste(patterns, collapse = ' OR '), 'in', dir)) # nolint # nolint
    hit[1]
}

mtx_path <- find_one(visium_dir, c('matrix\\.mtx(\\.gz)?$'))
feat_path <- find_one(visium_dir, c('features\\.tsv(\\.gz)?$', 'genes\\.tsv(\\.gz)?$'))
bc_path <- find_one(visium_dir, c('barcodes\\.tsv(\\.gz)?$'))
spatial_dir <- file.path(visium_dir, 'spatial')
if (!dir.exists(spatial_dir)) stop(paste("Missing spatial/ folder in:", visium_dir), call. = FALSE)

counts <- ReadMtx(mtx = mtx_path, features = feat_path, cells = bc_path, feature.column = 2)

img <- Read10X_Image(image.dir = spatial_dir, filter.matrix = FALSE)

counts_bcs <- colnames(counts)
img_bcs <- rownames(GetTissueCoordinates(img))

if (any(grepl('-1$', counts_bcs)) && !any(grepl('-1$', img_bcs))) {
    rownames(img@coordinates) <- paste0(img_bcs, '-1')
    img_bcs <- rownames(img@coordinates)
}

if (!any(grepl('-1$', counts_bcs)) && any(grepl('-1$', img_bcs))) {
    rownames(img@coordinates) <- sub('-1$', '', img_bcs)
    img_bcs <- rownames(img@coordinates)
}

common <- intersect(counts_bcs, img_bcs)
cat("counts cells:", length(counts_bcs), "\n")
cat("img cells:", length(img_bcs), "\n")
cat("overlap:", length(common), "\n")

if (length(common) == 0) {
    stop("No overlapping barcodes between counts and spatial coordinates. Check barcode suffix / tissue_positions.csv.", call.=FALSE)
}

counts2 <- counts[, common, drop = FALSE]

obj <- CreateSeuratObject(counts = counts2, assay = 'Spatial')

img2 <- img[common, ]

obj[['slice1']] <- img2

obj[['Spatial']] <- CreateAssayObject(counts = counts2)

DefaultAssay(obj) <- 'Spatial'

obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, selection.method = 'vst', nfeatures = 3000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, npcs = 50)

obj <- FindNeighbors(obj, dims = 1:30)
obj <- FindClusters(obj, resolution = 0.5)

obj <- RunUMAP(obj, dims = 1:30)
obj <- RunTSNE(obj, dims = 1:30)

p_clusters <- DimPlot(obj, reduction = 'umap', group.by = 'seurat_clusters', label = TRUE) + ggtitle('UMAP: Seurat Clusters')
p_sp_clusters <- SpatialDimPlot(obj, group.by = 'seurat_clusters', label = TRUE, label.size = 3) + ggtitle('Spatial: Seurat Clusters')
ggsave(file.path(outdir, 'Clusters_umap_spatial.png'), p_clusters + p_sp_clusters, width = 14, height = 6)

ref <- celldex::HumanPrimaryCellAtlasData()

expr <- GetAssayData(obj, assay = 'Spatial', layer = 'counts')

pred_main <- SingleR(test = as.matrix(expr),
                    ref = ref,
                    labels = ref$label.main)

obj$SingleR_label_main <- pred_main$labels
obj$SingleR_score_main <- pred_main$scores[cbind(seq_len(ncol(expr)), match(pred_main$labels, colnames(pred_main$scores)))]

pred_fine <- SingleR(test = as.matrix(expr),
                    ref = ref,
                    labels = ref$label.fine)
obj$SingleR_label_fine <- pred_fine$labels

p1 <- DimPlot(obj, reduction = 'umap', group.by = 'SingleR_label_main', label = TRUE, repel = TRUE) + ggtitle('SingleR (HPCA) label.main - UMAP')
p2 <- DimPlot(obj, reduction = 'tsne', group.by = 'SingleR_label_main', label = TRUE, repel = TRUE) + ggtitle('SingleR (HPCA) label.main - tSNE')
ggsave(file.path(outdir, 'SingleR_main_umap_tsne.png'), p1 + p2, width = 14, height = 6)

p3 <- SpatialDimPlot(obj, group.by = 'SingleR_label_main', label = TRUE, repel = TRUE, label.size = 3) + ggtitle('SingleR (HPCA) label.main - Spatial')
ggsave(file.path(outdir, 'SingleR_main_spatial.png'), p3, width = 8, height = 7)

p4 <- DimPlot(obj, reduction = 'umap', group.by = 'SingleR_label_fine', label = FALSE) + ggtitle('SingleR label.fine - UMAP')
ggsave(file.path(outdir, 'SingleR_fine_umap.png'), p4, width = 10, height = 7)

meta <- obj@meta.data
meta$barcode <- rownames(meta)
write.csv(meta, file.path(outdir, 'celltype_metadata_singleR.csv'), row.names = FALSE)

saveRDS(obj, file.path(outdir, 'spatial_seurat_singleR.rds'))

message('[DONE] SingleR outputs saved to: ', outdir)