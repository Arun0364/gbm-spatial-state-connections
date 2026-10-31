# Imports
import argparse
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import os

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCA, t-SNE, and UMAP pipeline on one or more AnnData .h5ad files"
    )

    # Required arguments
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to input .h5ad files (e.g. ST1.h5ad ST2.h5ad)"
    )
    parser.add_argument(
        "--sample-names",
        nargs="+",
        required=True,
        help="Sample names (same order as inputs)"
    )    
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for figures and processed h5ad files"
    )

    # Pipeline hyperparameters
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--n-comps-pca", type=int, default=50)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--tsne-perplexity", type=int, default=30)

    # Flags
    parser.add_argument(
        "--no-regress",
        action="store_true",
        help="Disable regression of technical covariates (total_counts, pct_counts_mt)"
    )
    parser.add_argument(
        "--no-save-h5ad",
        action="store_true",
        help="Do not save processed .h5ad outputs"
    )

    args = parser.parse_args()

    if len(args.inputs) != len(args.sample_names):
        raise ValueError("Number of --inputs must match number of --sample-names")
    
    return args
    
# Dimensionality reduction and Visualization function
def runDimredPipeline(
        annData,
        sample_name: str,
        out_dir: str,
        min_genes: int = 200,
        min_cells: int = 3,
        n_hvg: int = 3000,
        n_comps_pca: int = 50,
        n_pcs: int = 30,
        n_neighbors: int = 15,
        tsne_perplexity: int = 30,
        do_regress: bool = True,
        save_h5ad: bool = True,
        marker_genes: list[str] | None = None
):
    """
    Perform dimensionality reduction and visualization on an AnnData object.

    Parameters:
    annData: AnnData
        The input AnnData object.
    sample_name: str
        Name of the sample for labeling plots.
    out_dir: str
        Directory to save output plots and data.
    min_genes: int
        Minimum number of genes per cell for filtering.
    min_cells: int
        Minimum number of cells per gene for filtering.
    n_hvg: int
        Number of highly variable genes to select.
    n_pcs: int
        Number of principal components to compute.
    n_neighbors: int
        Number of neighbors for UMAP.
    tsne_perplexity: int
        Perplexity parameter for t-SNE.
    do_regress: bool
        Whether to regress out total counts and mitochondrial percentage.
    save_h5ad: bool
        Whether to save the processed AnnData object to an h5ad file.
    marker_genes: list[str] | None
        List of marker genes to highlight in plots.
    """

    annData_processed = annData.copy()
    os.makedirs(out_dir, exist_ok=True)
    sample_dir = os.path.join(out_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    # Tagging sample
    annData_processed.obs['sample_name'] = sample_name

    # Using counts as layer to X
    if "counts" in annData_processed.layers:
        annData_processed.X = annData_processed.layers['counts'].copy()
    
    # QC metrics
    if "mt" in annData_processed.var.columns:
        sc.pp.calculate_qc_metrics(annData_processed, qc_vars=["mt"], inplace=True)

    # Filtering
    sc.pp.filter_cells(annData_processed, min_genes=min_genes)
    sc.pp.filter_genes(annData_processed, min_cells=min_cells)

    # HVGs
    sc.pp.highly_variable_genes(annData_processed, n_top_genes=n_hvg, flavor="seurat_v3", subset=True)

    # Normalization + log1p
    sc.pp.normalize_total(annData_processed, target_sum=1e4)

    annData_processed.layers["norm_counts"] = annData_processed.X.copy()
    annData_processed.raw = annData_processed

    # Regression
    if do_regress:
        keys = [k for k in ["total_counts", "pct_counts_mt"] if k in annData_processed.obs.columns]
        if keys:
            if sp.issparse(annData_processed.X):
                annData_processed.X = np.asarray(annData_processed.X.todense(), order="C")
            else:
                annData_processed.X = np.ascontiguousarray(annData_processed.X)
            sc.pp.regress_out(annData_processed, keys)
    
    # Scaling
    sc.pp.scale(annData_processed, max_value=10)

    # PCA
    annData_pca = annData.copy()

    if "counts" in annData_pca.layers:
        annData_pca.X = annData_pca.layers["counts"].copy()

    annData_pca = annData_pca[annData_processed.obs_names, annData_processed.var_names].copy()

    sc.pp.normalize_total(annData_pca, target_sum=1e4)

    sc.tl.pca(annData_pca, n_comps=n_comps_pca, svd_solver="arpack")

    annData_processed.obsm["X_pca"] = annData_pca.obsm["X_pca"].copy()
    if "PCs" in annData_pca.varm:
        annData_processed.varm["PCs"] = annData_pca.varm["PCs"].copy()
    if "pca" in annData_pca.uns:
        annData_processed.uns["pca"] = annData_pca.uns["pca"].copy()

    annData_processed.uns["pca_input"] = "normalize_total(counts), no log1p"

    # Standard plots for PCA
    sc.pl.pca_variance_ratio(annData_processed, log=False, n_pcs=n_comps_pca, save=f"_{sample_name}.png", show=False)
    color_fields = [c for c in ["sample_name", "total_counts", "pct_counts_mt", "n_genes_by_counts"] if c in annData_processed.obs.columns]
    sc.pl.pca(annData_processed, color=color_fields[:3] if len(color_fields) else None, legend_loc="best", 
              save=f"_{sample_name}.png", show=False)
    sc.pl.pca_loadings(annData_processed, components=[1,2,3], n_points=30, save=f"_{sample_name}.png", show=False)

    # t-SNE
    sc.tl.tsne(annData_processed, 
               use_rep="X_pca",
                n_pcs=n_pcs,
                n_jobs=1,
                random_state=0,
                perplexity=tsne_perplexity)
    sc.pl.tsne(annData_processed, color=color_fields[:3] if len(color_fields) else None, legend_loc="best", 
               save=f"_{sample_name}.png", show=False)

    # Neighbors + UMAP
    sc.pp.neighbors(annData_processed, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.tl.umap(annData_processed)
    sc.pl.umap(annData_processed, color=color_fields[:3] if len(color_fields) else None, legend_loc="best",
               save = f"_{sample_name}.png", show=False)

    # Marker gene UMAPs
    if marker_genes:
        present_genes = [gene for gene in marker_genes if gene in annData_processed.var_names]
        missing_genes = [gene for gene in marker_genes if gene not in annData_processed.var_names]

        print(f"[{sample_name}] Marker genes present: {len(present_genes)/len(marker_genes):.2%}")

        if missing_genes:
            print(f"[{sample_name}] Missing marker genes: {', '.join(missing_genes)}")

        for i in range(0, len(present_genes), 6):
            batch = present_genes[i:i+6]
            sc.pl.umap(annData_processed,
                       color=batch,
                       legend_loc="best",
                       use_raw=True,
                       ncols=3,
                       save=f"_{sample_name}_markers_{i//6 + 1}.png",
                       show=False)
    
    # Move figures to sample directory
    if os.path.isdir("Figures"):
        for fn in os.listdir("Figures"):
            if fn.endswith(".png") and f"_{sample_name}_" in fn:
                os.replace(os.path.join("Figures", fn), os.path.join(sample_dir, fn))

    # Save processed AnnData
    if save_h5ad:
        out_h5ad = os.path.join(sample_dir, f"{sample_name}_pca_tsne_umap.h5ad")
        annData_processed.write_h5ad(out_h5ad)
        print(f"[{sample_name}] Processed AnnData saved to {out_h5ad}")
    
    return annData_processed

# Main function
def main(args):
    markers = [
        # Neuronal
        "SNAP25", "SYT1", "GRIA1",
        # Oligodendro / myelin
        "MBP", "PLP1", "MOG", "OLIG1", "OLIG2",
        # Astro
        "GFAP", "AQP4", "ALDH1L1",
        # Microglia / macrophage
        "C1QA", "C1QB", "LST1", "TYROBP",
        # T cell
        "CD3D", "CD3E", "TRBC1",
        # Tumor / reactive
        "EGFR", "CD44", "VIM", "CHI3L1", "SPP1",
        # Hypoxia / angiogenesis
        "VEGFA", "HIF1A", "CA9"
        ]
    
    for path, sample in zip(args.inputs, args.sample_names):
        print(f"\n>>> Processing {sample}")
        annData = sc.read_h5ad(path)

        runDimredPipeline(
            annData,
            sample_name=sample,
            out_dir=args.out_dir,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            n_hvg=args.n_hvg,
            n_comps_pca=args.n_comps_pca,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            tsne_perplexity=args.tsne_perplexity,
            do_regress=not args.no_regress,
            save_h5ad=not args.no_save_h5ad,
            marker_genes=markers
        )

        print(f">>> Finished {sample}\n")
    
    print("All samples processed successfully.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
