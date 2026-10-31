from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

def ensure_figdir(sample_dir: Path):
    sample_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(sample_dir)

def get_W(aData: sc.AnnData, key: str = "X_W") -> np.ndarray:
    if key not in aData.obsm:
        raise KeyError(f"Missing adata.obsm['{key}']. Run AA first and store W.")
    W = np.asarray(aData.obsm[key], dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be 2D (n_obs, K).")
    return W

def add_entropy_and_purity(aData: sc.AnnData, W_key: str = "X_W", eps: float = 1e-12):
    W = get_W(aData, W_key)

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Wn = W /row_sums

    entropy = -np.sum(Wn * np.log(Wn + eps), axis=1)
    aData.obs["W_entropy"] = entropy.astype(float)

    if 'maxW' not in aData.obs:
        aData.obs['maxW'] = np.max(Wn, axis=1).astype(float)
    if 'argmax' not in aData.obs:
        aData.obs['argmax'] = Wn.argmax(axis=1).astype(int)

def label_scs_by_quantile(aData: sc.AnnData, q: float = 0.90):
    if 'maxW' not in aData.obs:
        raise KeyError("Need adata.obs['maxW'] (run add_entropy_and_purity first).")
    cutoff = float(np.quantile(aData.obs['maxW'].to_numpy(), q))
    aData.obs['spot_type'] = pd.Categorical(
        np.where(aData.obs['maxW'].to_numpy() >= cutoff, 'SCS', 'Bulk')
    )
    aData.uns.setdefault('aa_qc', {})
    aData.uns['aa_qc'].update({'scs_quantile': q, 'maxW_cutoff': cutoff})
    return cutoff

def save_basic_qc_plots(
        aData: sc.AnnData,
        sample_dir: Path,
        sample_name: str,
        spot_size: float = 2.3,
        alpha_img: float = 0.15
):
    ensure_figdir(sample_dir)
    if 'X_umap' not in aData.obsm:
        if 'X_pca' in aData.obsm:
            sc.pp.neighbors(aData, n_pcs=min(30, aData.obsm["X_pca"].shape[1]), n_neighbors=15)
        else:
            sc.pp.neighbors(aData, n_neighbors=15)
        sc.tl.umap(aData)

    for col in ['spot_type', 'maxW', 'W_entropy', 'argmax']:
        if col in aData.obs:
            sc.pl.umap(aData, color=col, show=False, save=f"_{sample_name}_umap_{col}.png")
    
    for col in ['spot_type', 'maxW', 'W_entropy', 'argmax']:
        if col in aData.obs:
            try:
                sc.pl.spatial(
                    aData,
                    color=col,
                    spot_size=spot_size,
                    alpha_img=alpha_img,
                    show=False,
                    save=f"_{sample_name}_spatial_{col}.png"
                )
            except Exception:
                sc.pl.spatial(
                    aData,
                    color=col,
                    img_key=None,
                    spot_size=spot_size,
                    show=False,
                    save=f"_{sample_name}_spatial_{col}_nohist.png"
                )

def save_per_archetype_spatial_weights(
        aData: sc.AnnData,
        sample_dir: Path,
        sample_name: str,
        W_key: str = "X_W",
        spot_size: float = 2.3,
        alpha_img: float = 0.15
):
    ensure_figdir(sample_dir)
    W = get_W(aData, W_key)
    K = W.shape[1]

    for k in range(K):
        aData.obs[f"W{k}"] = W[:, k].astype(float)

    for k in range(K):
        col = f"W{k}"
        try:
            sc.pl.spatial(
                aData,
                color=col,
                spot_size=spot_size,
                alpha_img=alpha_img,
                show=False,
                save=f"_{sample_name}_spatial_{col}.png"
            )
        except Exception:
            sc.pl.spatial(
                aData,
                color=col,
                img_key=None,
                spot_size=spot_size,
                show=False,
                save=f"_{sample_name}_spatial_{col}_nohist.png"
            )

def markers_by_argmax(
        aData: sc.AnnData,
        groupby: str = 'argmax',
        method: str = 'wilcoxon',
        n_top: int = 50
) -> pd.DataFrame:
    if groupby not in aData.obs:
        raise KeyError(f"Missing adata.obs['{groupby}'].")
    
    aData.obs[groupby] = pd.Categorical(aData.obs[groupby].astype(str))

    sc.tl.rank_genes_groups(aData, groupby=groupby, method=method)
    res = aData.uns['rank_genes_groups']

    groups = res['names'].dtype.names
    rows = []
    for g in groups:
        names = res['names'][g][:n_top]
        scores = res['scores'][g][:n_top] if 'scores' in res else [np.nan] * len(names)
        logfc = res['logfoldchanges'][g][:n_top] if 'logfoldchanges' in res else [np.nan] * len(names)
        pvals = res['pvals_adj'][g][:n_top] if 'pvals_adj' in res else [np.nan] * len(names)
        for i, gene in enumerate(names):
            rows.append(
                {
                    "group": g,
                    "rank": i + 1,
                    "gene": gene,
                    "score": float(scores[i]) if i < len(scores) else np.nan,
                    "logFC": float(logfc[i]) if i < len(logfc) else np.nan,
                    "p_adj": float(pvals[i]) if i < len(pvals) else np.nan
                }
            )
    return pd.DataFrame(rows)
    
def markers_topW_vs_rest(
        aData: sc.AnnData,
        W_key: str = 'X_W',
        top_frac: float = 0.10,
        method: str = 'wilcoxon',
        n_top: int = 50
) -> pd.DataFrame:
    W = get_W(aData, W_key)
    K = W.shape[1]
    n = aData.n_obs
    top_n = max(10, int(np.ceil(top_frac * n)))

    rows = []
    for k in range(K):
        idx_sorted = np.argsort(-W[:, k])
        top_idx = idx_sorted[:top_n]
        group = np.array(['rest'] * n, dtype=object)
        group[top_idx] = f"A{k}_top"
        aData.obs['aa_top_group'] = pd.Categorical(group)

        sc.tl.rank_genes_groups(aData, groupby='aa_top_group', groups=[f'A{k}_top'], reference='rest', method=method)
        res = aData.uns['rank_genes_groups']

        names = res["names"][f"A{k}_top"][:n_top]
        scores = res["scores"][f"A{k}_top"][:n_top] if "scores" in res else [np.nan] * len(names)
        logfc = res["logfoldchanges"][f"A{k}_top"][:n_top] if "logfoldchanges" in res else [np.nan] * len(names)
        pvals = res["pvals_adj"][f"A{k}_top"][:n_top] if "pvals_adj" in res else [np.nan] * len(names)

        for i, gene in enumerate(names):
            rows.append(
                {
                    'archetype': k,
                    'comparison': f'top{top_frac:.2f}_vs_rest',
                    'rank': i + 1,
                    'gene': gene,
                    'score': float(scores[i]) if i < len(scores) else np.nan,
                    'logFC': float(logfc[i]) if i < len(logfc) else np.nan,
                    'p_adj': float(pvals[i]) if i < len(pvals) else np.nan
                }
            )

    if 'aa_top_group' in aData.obs:
        del aData.obs['aa_top_group']

    return pd.DataFrame(rows)

def _get_counts_matrix(aData: sc.AnnData) -> np.ndarray:
    if 'counts' in aData.layers:
        X = aData.layers['counts']
    else:
        X = aData.X
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    return X.astype(float)

def pseudo_bulk_by_knn(
        aData: sc.AnnData,
        W_key: str = 'X_W',
        k_neighbors: int = 15,
        n_pseudobulk: int = 300,
        seed: int = 0,
        use_bulk_only: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'spatial' not in aData.obsm:
        raise KeyError("Missing adata.obsm['spatial'] required for kNN neighborhoods.")
    coords = np.asarray(aData.obsm['spatial'], dtype=float)
    
    W = get_W(aData, W_key)
    X = _get_counts_matrix(aData)

    obs = aData.obs.copy()
    if use_bulk_only:
        if 'spot_type' not in obs:
            raise KeyError("spot_type missing. Run label_scs_by_quantile first or set use_bulk_only=False.")
        keep = (obs['spot_type'].astype(str) == 'Bulk').to_numpy()
    else:
        keep = np.ones(aData.n_obs, dtype=bool)

    coords = coords[keep]
    W = W[keep]
    X = X[keep]
    idx_map = np.where(keep)[0]

    rng = np.random.default_rng(seed)
    anchors = rng.choice(coords.shape[0], size=min(n_pseudobulk, coords.shape[0]), replace=False)

    gene_names = aData.var_names.tolist()
    K = W.shape[1]

    expr_rows = []
    prop_rows = []

    for j, a in enumerate(anchors):
        d2 = np.sum((coords-coords[a])**2, axis=1)
        nn = np.argsort(d2)[:k_neighbors]

        expr = X[nn].sum(axis=0)
        prop = W[nn].mean(axis=0)

        gid = f"pb_{j:04d}_anchor{idx_map[a]}"
        expr_rows.append(pd.Series(expr, index=gene_names, name=gid))
        prop_rows.append(pd.Series(prop, index=[f"A{k}" for k in range(K)], name=gid))

    expr_df = pd.DataFrame(expr_rows)
    prop_df = pd.DataFrame(prop_rows)
    return expr_df, prop_df

def run_next_steps(
        input_h5ad: str | Path,
        out_dir: str | Path,
        sample_name: str,
        scs_quantile: float = 0.90,
        make_plots: bool = True,
        make_weight_maps: bool = True,
        make_markers: bool = True,
        marker_mode: str = 'argmax',
        topW_frac: float = 0.10,
        make_pseudobulk: bool = True,
        pseudobulk_mode: str = 'knn'
):
    input_h5ad = Path(input_h5ad)
    out_dir = Path(out_dir)
    sample_dir = out_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    aData = sc.read_h5ad(input_h5ad)

    add_entropy_and_purity(aData)
    cutoff = label_scs_by_quantile(aData, scs_quantile)

    if make_plots:
        save_basic_qc_plots(aData, sample_dir, sample_name)

    if make_weight_maps:
        save_per_archetype_spatial_weights(aData, sample_dir, sample_name)

    if make_markers:
        if 'counts' in aData.layers:
            a_deg = aData.copy()
            a_deg.X = a_deg.layers['counts'].copy()
            sc.pp.normalize_total(a_deg, target_sum=1e4)
        else:
            a_deg = aData.copy()

        if marker_mode == 'argmax':
            df_markers = markers_by_argmax(a_deg, groupby='argmax', n_top=50)
        elif marker_mode == 'topW':
            df_markers = markers_topW_vs_rest(a_deg, W_key='X_W', top_frac=topW_frac, n_top=50)
        else:
            raise ValueError("marker_mode must be 'argmax' or 'topW'")
        
        df_markers.to_csv(sample_dir / f"{sample_name}_archetype_markers.csv", index=False)

    if make_pseudobulk:
        if pseudobulk_mode == 'knn':
            expr_df, prop_df = pseudo_bulk_by_knn(aData, W_key='X_W', k_neighbors=15, n_pseudobulk=300, seed=0, use_bulk_only=True)
        else:
            raise ValueError("pseudobulk_mode must be 'knn'")
        
        expr_df.to_csv(sample_dir / f'{sample_name}_pseudo_bulk_expression.csv')
        prop_df.to_csv(sample_dir / f'{sample_name}_pseudo_bulk_proportions.csv')

    aData.uns.setdefault('aa_qc', {})
    aData.uns['aa_qc'].update({'maxW_cutoff': cutoff})
    aData.write_h5ad(sample_dir / f"{sample_name}_ns_annotated.h5ad")

    print(f'[{sample_name}] Done.')
    print(f"  - SCS cutoff (q={scs_quantile}): maxW >= {cutoff:.4f}")
    print(f"  - Output folder: {sample_dir}")

for sample in ["G740_ST1", "G4_ST2"]:
    run_next_steps(
        input_h5ad=f"/Users/arun/Desktop/Research/Results/PostAA_tumor_unmix_k4_markers/{sample}/{sample}_annotated.h5ad",
        out_dir="/Users/arun/Desktop/Research/Results/PostAA_tumor_unmix_k4_markers",
        sample_name=sample,
        scs_quantile=0.90,
        make_plots=True,
        make_weight_maps=True,
        make_markers=True,
        marker_mode="argmax",
        topW_frac=0.10,
        make_pseudobulk=True,
        pseudobulk_mode="knn"
    )