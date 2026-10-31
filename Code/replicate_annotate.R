suppressPackageStartupMessages({
    library(Seurat)
    library(Matrix)
    library(SingleR)
    library(celldex)
    library(future)
})
plan(sequential)
options(future.globals.maxSize = 8 * 1024^3)
args <- commandArgs(trailingOnly = TRUE)
section_dir <- args[1]
section <- args[2]
code_dir <- "/Users/arun/Desktop/Research/Code"
msg <- function(...) cat(sprintf(...), "\n")

exprs <- parse(file.path(code_dir, "Neftel_States_Visium_Pipeline.R"))
for (e in exprs) {
    if (is.call(e) && identical(e[[1]], as.name("<-")) &&
        as.character(e[[2]]) %in% c("neftel_signatures", "MIN_OVERLAP", "score_neftel_curated")) eval(e)
}

mtx_dir <- file.path(section_dir, "counts_mtx")
counts <- ReadMtx(mtx = file.path(mtx_dir, "matrix.mtx"), features = file.path(mtx_dir, "features.tsv"),
                  cells = file.path(mtx_dir, "barcodes.tsv"), feature.column = 2)
obj <- CreateSeuratObject(counts = counts, assay = "Spatial", project = section)
DefaultAssay(obj) <- "Spatial"
obj <- NormalizeData(obj, verbose = FALSE)

ref <- celldex::HumanPrimaryCellAtlasData()
expr <- GetAssayData(obj, assay = "Spatial", layer = "counts")
pred_main <- SingleR(test = as.matrix(expr), ref = ref, labels = ref$label.main)
obj$SingleR_label_main <- pred_main$labels
obj$SingleR_score_main <- pred_main$scores[cbind(seq_len(ncol(expr)), match(pred_main$labels, colnames(pred_main$scores)))]
pred_fine <- SingleR(test = as.matrix(expr), ref = ref, labels = ref$label.fine)
obj$SingleR_label_fine <- pred_fine$labels
meta <- obj@meta.data
meta$barcode <- rownames(meta)
write.csv(meta, file.path(section_dir, "celltype_metadata_singleR.csv"), row.names = FALSE)
msg("[singleR] %s: %d spots", section, ncol(obj))
print(sort(table(obj$SingleR_label_main), decreasing = TRUE))

obj <- score_neftel_curated(obj, neftel_signatures)
md <- obj@meta.data
md$barcode <- rownames(md)
write.csv(md, file.path(section_dir, "neftel_curated.csv"), row.names = FALSE)
print(table(obj$Neftel_state))