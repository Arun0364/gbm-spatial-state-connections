suppressPackageStartupMessages(({
    library(Seurat)
    library(Matrix)
    library(ggplot2)
    library(patchwork)
    library(pheatmap)
    library(future)
}))

plan(sequential)
options(future.globals.maxSize = 8 * 1024^3)

ST1_DIR <- '/Users/arun/Desktop/Research/Data/G740_ST1'
ST2_DIR <- '/Users/arun/Desktop/Research/Data/G4_ST2'

OUTDIR <- '/Users/arun/Desktop/Research/Results/neftel_outputs_curated'
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

NDIMS <- 30
MIN_OVERLAP <- 10

msg <- function(...) cat(sprintf(...), '\n')

neftel_signatures <- list(
     AC_like = c(
    "GFAP","AQP4","ALDOC","SLC1A3","SOX9","CLU","S100B","APOE","FABP7","HOPX",
    "HEPACAM","SPARCL1","ID3","ID4","SLC4A4","PLP1","MT3","LCN2","GJA1","VIM"
  ),
  MES_like = c(
    "CHI3L1","CCL2","CXCL8","IL1B","TNFAIP3","NFKBIA","SERPINE1","CD44","ANXA1","LGALS3",
    "IFITM3","S100A10","S100A11","S100A6","TIMP1","CTSS","LST1","TYMP","EMP1","FOS"
  ),
  OPC_like = c(
    "PDGFRA","CSPG4","OLIG1","OLIG2","SOX10","NKX2-2","VCAN","DLL3","ASCL1","BCAN",
    "PLP1","MBP","MOG","MAG","GPR17","ERMN","UGT8","MOBP","CLDN11","TF"
  ),
  NPC_like = c(
    "SOX4","SOX11","DCX","TUBB3","STMN2","NEUROD1","NEUROD6","ASCL1","DLL1","HES6",
    "ELAVL4","ENO2","SYT1","SNAP25","MAP2","PAX6","NR2E1","MKI67","TOP2A","HMGB2"
  )
)

find_one <- function(dir, patterns) {
  files <- list.files(dir, full.names = TRUE)
  hit <- unlist(lapply(patterns, function(p) grep(p, files, value = TRUE, ignore.case = TRUE)))
  if (length(hit) == 0) stop(paste("Could not find:", paste(patterns, collapse = " OR "), "in", dir))
  hit[1]
}

load_visium_any <- function(visium_dir, sample_name) {
    spatial_dir <- file.path(visium_dir, "spatial")
    if (!dir.exists(spatial_dir)) stop(paste("Missing spatial/ folder in:", visium_dir))

    mtx_path  <- find_one(visium_dir, c("matrix\\.mtx(\\.gz)?$"))
    feat_path <- find_one(visium_dir, c("features\\.tsv(\\.gz)?$", "genes\\.tsv(\\.gz)?$"))
    bc_path   <- find_one(visium_dir, c("barcodes\\.tsv(\\.gz)?$"))
    
    counts <- ReadMtx(mtx = mtx_path, features = feat_path, cells = bc_path, feature.column = 2)
    
    obj <- CreateSeuratObject(counts = counts, assay = "Spatial", project = sample_name)
    DefaultAssay(obj) <- "Spatial"
    obj$dataset <- sample_name
    
    img <- Read10X_Image(image.dir = spatial_dir)

    coords <- GetTissueCoordinates(img)
    img_bcs <- rownames(coords)

    obj_bcs <- colnames(obj)

    strip <- function(x) sub('-1$', '', x)
    obj_key <- strip(obj_bcs)
    img_key <- strip(img_bcs)

    common_key <- intersect(obj_key, img_key)
    msg("[visium] %s: counts=%d | img=%d | overlap(keys)=%d",
    sample_name, length(obj_bcs), length(img_bcs), length(common_key))

    if (length(common_key) == 0) {
        msg("[debug] Example count barcodes: %s", paste(head(obj_bcs, 5), collapse=", "))
        msg("[debug] Example image barcodes: %s", paste(head(img_bcs, 5), collapse=", "))
        stop("No overlapping barcodes between matrix and image (even after normalization).")
    }

    keep_obj <- obj_bcs[obj_key %in% common_key]
    keep_img <- img_bcs[img_key %in% common_key]

    obj <- subset(obj, cells = keep_obj)

    coords2 <- coords[keep_img, , drop = FALSE]
    img <- img[keep_img, ]

    img <- img[, colnames(obj)]

    obj[['slice1']] <- img
    return(obj)
}

score_neftel_curated <- function(sp, signatures) {
    DefaultAssay(sp) <- 'Spatial'
    sp <- SCTransform(sp, assay = 'Spatial', verbose = FALSE)

    sig_sp <- lapply(signatures, function(g) intersect(unique(g), rownames(sp)))
    overlap_sizes <- sapply(sig_sp, length)
    msg("[sig] overlap counts:"); print(overlap_sizes)

    keep <- overlap_sizes >= MIN_OVERLAP
    if (!any(keep)) stop("No signatures have enough overlap with Visium genes.")
    sig_sp <- sig_sp[keep]
    state_names <- names(sig_sp)

    sp <- AddModuleScore(sp, features = sig_sp, assay = 'SCT', name = 'NeftelProg_')

    score_cols <- paste0('NeftelProg_', seq_along(state_names))
    score_mat <- sp@meta.data[, score_cols, drop = FALSE]
    colnames(score_mat) <- state_names

    best_idx <- apply(score_mat, 1, which.max)
    sp$Neftel_state <- factor(state_names[best_idx], levels = state_names)

    sorted_scores <- t(apply(score_mat, 1, function(x) sort(x, decreasing = TRUE)))
    sp$Neftel_confidence <- sorted_scores[, 1] - sorted_scores[, 2]

    return(sp)
}

make_plots <- function(sp, sample_name, outdir) {
    dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

    sp <- RunPCA(sp, assay = 'SCT', verbose = FALSE)
    sp <- RunUMAP(sp, dims = 1:NDIMS, verbose = FALSE)
    sp <- RunTSNE(sp, dims = 1:NDIMS, verbose = FALSE)

    p_umap_state <- DimPlot(sp, reduction = 'umap', group.by = 'Neftel_state', label = TRUE) + ggtitle(paste0(sample_name, ': UMAP (Neftel curated)'))
    p_tsne_state <- DimPlot(sp, reduction = 'tsne', group.by = 'Neftel_state', label = TRUE) + ggtitle(paste0(sample_name, ': tSNE (Neftel curated)'))

    ggsave(file.path(outdir, paste0(sample_name, '_UMAP_Neftel.png')), p_umap_state, width = 7, height = 5, dpi = 200)
    ggsave(file.path(outdir, paste0(sample_name, '_tSNE_Neftel.png')), p_tsne_state, width = 7, height = 5, dpi = 200)

    p_spatial_state <- SpatialDimPlot(sp, group.by = 'Neftel_state') + ggtitle(paste0(sample_name, ': Spatial (Neftel curated)'))
    ggsave(file.path(outdir, paste0(sample_name, '_Spatial_Neftel.png')), p_spatial_state, width = 7, height = 5, dpi = 200)

    p_conf <- SpatialFeaturePlot(sp, features = 'Neftel_confidence') + ggtitle(paste0(sample_name, ": Spatial confidence (top1-top2)"))
    ggsave(file.path(outdir, paste0(sample_name, "_Spatial_Neftel_conf.png")), p_conf, width = 7, height = 5, dpi = 200)

    if ('archetype' %in% colnames(sp@meta.data)) {
        p_umap_arch <- DimPlot(sp, reduction = 'umap', group.by = 'archetype', label = TRUE) + ggtitle(paste0(sample_name, ': UMAP (Archetype)'))
        p_spatial_arch <- SpatialDimPlot(sp, group.by = 'archetype') + ggtitle(paste0(sample_name, ': Spatial (Archetype)'))

        ggsave(file.path(outdir, paste0(sample_name, '_UMAP_Archetype.png')), p_umap_arch, width = 7, height = 5, dpi = 200)
        ggsave(file.path(outdir, paste0(sample_name, '_Spatial_Archetype.png')), p_spatial_arch, width = 7, height = 5, dpi = 200)

        tab <- prop.table(table(sp$archetype, sp$Neftel_state), margin = 1)
        tab <- as.matrix(tab)
        pheatmap(tab,
            cluster_rows = FALSE, cluster_cols = FALSE,
            main = paste0(sample_name, ': Neftel state proportions per archetype'),
            filename = file.path(outdir, paste0(sample_name, '_Archetype_Neftel_heatmap.png')),
            width = 7, height = 5)
            }
            saveRDS(sp, file.path(outdir, paste0(sample_name, "_visium_neftel_scored.rds")))
            sp
}

datasets <- list(G740_ST1 = ST1_DIR, G4_ST2 = ST2_DIR)
all_scored <- list()

for (nm in names(datasets)) {
    vis_dir <- datasets[[nm]]
    if (!dir.exists(vis_dir)) {
        msg("[skip] Missing dir: %s", vis_dir)
        next
    }

    msg("\n=== Processing %s ===", nm)
    sp <- load_visium_any(vis_dir, sample_name = nm)
    sp <- score_neftel_curated(sp, neftel_signatures)

    ds_out <- file.path(OUTDIR, nm)
    sp <- make_plots(sp, nm, ds_out)

    all_scored[[nm]] <- sp
}

msg("\n=== Combined summary plots (ST1 vs ST2) ===")

if (length(all_scored) >= 1) {
    meta_list <- lapply(names(all_scored), function(nm) {
        md <- all_scored[[nm]]@meta.data
        md$dataset <- nm
        md$barcode <- rownames(md)
        md
    })
    meta_all <- do.call(rbind, meta_list)
    write.csv(meta_all, file.path(OUTDIR, 'visium_all_spots_neftel_curated.csv'), row.names = FALSE)

    meta_all <- meta_all[!is.na(meta_all$Neftel_state), , drop=FALSE]
    tab_ds <- prop.table(table(meta_all$dataset, meta_all$Neftel_state), margin = 1)

    df_tab <- as.data.frame(tab_ds)
    colnames(df_tab) <- c('dataset', 'Neftel_state', 'prop')

    p_bar <- ggplot(df_tab, aes(x = Neftel_state, y = prop, fill = dataset)) +
        geom_col(position = 'dodge') + theme_bw() +
        ggtitle('Curated Neftel state proportions per dataset (Visium spots)')
    ggsave(file.path(OUTDIR, 'ST1_ST2_Neftel_state_proportions.png'), p_bar, width = 9, height = 4, dpi = 200)

    pheatmap(
        as.matrix(tab_ds),
        cluster_rows = FALSE, cluster_cols = FALSE,
        main = 'Curated Neftel proportions (rows=datasets, cols=states)',
        filename = file.path(OUTDIR, 'ST1_ST2_Neftel_state_heatmap.png'),
        width = 7, height = 3
    )
}

msg('\n[DONE] Outputs saved in: %s', OUTDIR)