from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


def load_intervals(interval_pkl: str | Path) -> pd.DataFrame:
    interval_pkl = Path(interval_pkl)
    with open(interval_pkl, "rb") as f:
        d = pickle.load(f)

    intervals = d["OriginalSCSInterval"].copy()
    intervals["chromosome"] = intervals["chromosome"].astype(str).str.replace("^chr", "", regex=True)
    intervals = intervals.reset_index(drop=True)

    need = {"chromosome", "begin", "end"}
    if not need.issubset(intervals.columns):
        raise ValueError(f"intervals missing columns: {need - set(intervals.columns)}")
    if len(intervals) != 9934:
        print(f"[WARN] intervals count = {len(intervals)} (expected 9934)")
    return intervals


def normalize_gene_ids(idx: pd.Index | pd.Series) -> pd.Index:
    s = pd.Index(idx.astype(str))
    return s.str.replace(r"\..*$", "", regex=True)


def build_symbol_to_ensg(gene_coords: pd.DataFrame) -> pd.Series:
    if "gene_id" not in gene_coords.columns:
        raise ValueError("gene_coords missing gene_id")
    if "gene_name" not in gene_coords.columns:
        raise ValueError("gene_coords missing gene_name (symbol column)")

    df = gene_coords[["gene_name", "gene_id"]].dropna().copy()
    df["gene_id"] = df["gene_id"].astype(str).str.replace(r"\..*$", "", regex=True)
    df["gene_name"] = df["gene_name"].astype(str)

    df = df.drop_duplicates(subset=["gene_name"], keep="first")
    return pd.Series(df["gene_id"].values, index=df["gene_name"].values, name="gene_id")


def build_gene_to_interval_map(
    gene_coords: pd.DataFrame,
    intervals: pd.DataFrame,
    gene_id_col: str = "gene_id",
) -> pd.Series:
    need = {gene_id_col, "chromosome", "start", "end"}
    if not need.issubset(gene_coords.columns):
        raise ValueError(f"gene_coords missing columns: {need - set(gene_coords.columns)}")

    gc = gene_coords[[gene_id_col, "chromosome", "start", "end"]].dropna().copy()
    gc["chromosome"] = gc["chromosome"].astype(str).str.replace("^chr", "", regex=True)
    gc[gene_id_col] = gc[gene_id_col].astype(str).str.replace(r"\..*$", "", regex=True)
    gc["mid"] = ((gc["start"].astype(float) + gc["end"].astype(float)) / 2.0).astype(int)

    chr_to_int = {}
    for c, sub in intervals.groupby("chromosome", sort=False):
        sub = sub.sort_values("begin").reset_index()
        chr_to_int[c] = (sub["begin"].to_numpy(), sub["end"].to_numpy(), sub["index"].to_numpy())

    gene_to_interval = {}
    for _, row in gc.iterrows():
        g = row[gene_id_col]
        c = row["chromosome"]
        mid = int(row["mid"])
        if c not in chr_to_int:
            continue
        begins, ends, idxs = chr_to_int[c]
        j = np.searchsorted(begins, mid, side="right") - 1
        if j >= 0 and mid < ends[j]:
            gene_to_interval[g] = int(idxs[j])

    return pd.Series(gene_to_interval, name="interval_index")


def map_symbols_to_ensg_columns(X: pd.DataFrame, symbol_to_ensg: pd.Series) -> tuple[pd.DataFrame, int]:
    cols = pd.Index(X.columns.astype(str))
    mapped = cols.map(lambda s: symbol_to_ensg.get(s, np.nan))
    ok = pd.notna(mapped)
    n_mapped = int(ok.sum())

    X2 = X.loc[:, ok].copy()
    X2.columns = mapped[ok].astype(str)
    X2 = X2.T.groupby(level=0).mean(numeric_only=True).T
    return X2, n_mapped


def aggregate_genes_to_intervals_mean(
    X_ensg: pd.DataFrame,
    gene_to_interval: pd.Series,
    n_intervals: int,
    fill_value: float = 2.0,
) -> pd.DataFrame:
    X = X_ensg.copy()
    X.columns = normalize_gene_ids(X.columns)

    common = [g for g in X.columns if g in gene_to_interval.index]
    if len(common) == 0:
        raise ValueError("No ENSG columns in X overlap gene_to_interval. Check mapping.")

    Xg = X[common]
    int_idx = gene_to_interval.loc[common].astype(int).values

    Xt = Xg.T
    Xt["interval"] = int_idx
    Yt = Xt.groupby("interval").mean(numeric_only=True)
    Y = Yt.T

    Y = Y.reindex(columns=range(n_intervals))
    Y = Y.fillna(Y.mean(axis=0, skipna=True))
    Y = Y.fillna(fill_value)
    return Y


def build_cref_from_w_weighted_mean(
    annotated_h5ad: str | Path,
    target_genes: pd.Index,
    symbol_to_ensg: pd.Series,
    gene_to_interval: pd.Series,
    n_intervals: int,
    fill_value: float = 2.0,
    w_cols: list[str] | None = None,
) -> np.ndarray:
    if w_cols is None:
        w_cols = ["W0", "W1", "W2", "W3"]

    ad = sc.read_h5ad(str(annotated_h5ad))
    for col in w_cols:
        if col not in ad.obs.columns:
            raise ValueError(f"Missing W column '{col}' in h5ad obs")

    if "counts" in ad.layers:
        X = ad.layers["counts"]
    else:
        X = ad.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(np.float64)

    ad_genes = pd.Index(ad.var_names.astype(str))
    common_genes = [g for g in target_genes if g in ad_genes]
    if len(common_genes) == 0:
        raise ValueError("No overlap between target_genes and h5ad var_names")
    gene_idx = [ad_genes.get_loc(g) for g in common_genes]
    X_sub = X[:, gene_idx]

    W = ad.obs[w_cols].to_numpy(dtype=np.float64)

    ref_rows = []
    for k, col in enumerate(w_cols):
        w = W[:, k]
        w_sum = w.sum()
        if w_sum < 1e-12:
            print(f"[WARN] {col} weights sum near zero — falling back to uniform")
            w = np.ones(len(w), dtype=np.float64)
            w_sum = float(len(w))
        centroid = (X_sub.T @ w) / w_sum
        ref_rows.append(centroid)
        print(f"  [{col}] W-weighted centroid: mean={centroid.mean():.3f}  nonzero={np.count_nonzero(centroid)}")

    ref_df = pd.DataFrame(ref_rows, index=w_cols, columns=pd.Index(common_genes))
    ref_df_ensg, n_mapped = map_symbols_to_ensg_columns(ref_df, symbol_to_ensg)
    print(f"[CREF] {n_mapped}/{len(common_genes)} genes mapped to ENSG | after collapse: {ref_df_ensg.shape}")

    ref_interval = aggregate_genes_to_intervals_mean(
        ref_df_ensg, gene_to_interval, n_intervals=n_intervals, fill_value=fill_value
    )
    CRefer = ref_interval.T.to_numpy(dtype=float)
    return CRefer


def load_clone_state_map(state_labels_csv: str | Path) -> dict:
    df = pd.read_csv(state_labels_csv)
    return dict(zip(df["archetype"].astype(int), df["dominant_state"].astype(str)))


def write_mimo_pkl(
    out_pkl: Path,
    TumorSample: np.ndarray,
    CRefer: np.ndarray,
    FRefer: np.ndarray,
    ProbeIndex: list[int],
    clone_state_map: dict | None = None,
):
    K = CRefer.shape[1]
    n_bulk = TumorSample.shape[1]
    m = len(ProbeIndex)

    out = {
        "TumorSample": TumorSample,
        "CRefer": CRefer,
        "FRefer": FRefer,
        "ProbeIndex": ProbeIndex,
        "clone_state_map": clone_state_map or {},
        "uFISHRefer": np.full((m, K), 2.0, dtype=float),
        "FISHRefer": np.full((m, K), 2.0, dtype=float),
        "referFISHploidy": [2.0] * K,
        "ReferPloidy": [2.0] * K,
        "TruePloidy": [2.0] * K,
        "CInitial": CRefer.T.copy(),
        "COrigin": CRefer.copy(),
        "CTrue": CRefer.copy(),
        "FTrue": FRefer.copy(),
        "FTrueAll": np.zeros((K, n_bulk), dtype=float),
        "FISHcounts": [None] * n_bulk,
        "SimulatedFISH": np.full((m, K), 2.0, dtype=float),
    }

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-pkl",     required=True)
    ap.add_argument("--gene-coords",      required=True)
    ap.add_argument("--annotated-h5ad",   required=True)
    ap.add_argument("--pseudo-bulk-expr", required=True)
    ap.add_argument("--pseudo-bulk-prop", required=True)
    ap.add_argument("--state-labels-csv", required=True)
    ap.add_argument("--out-dir",          required=True)
    ap.add_argument("--n-cases",          type=int, default=10)
    ap.add_argument("--bulks-per-case",   type=int, default=30)
    ap.add_argument("--seed",             type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals = load_intervals(args.interval_pkl)
    n_intervals = len(intervals)

    gene_coords = pd.read_csv(args.gene_coords)
    symbol_to_ensg = build_symbol_to_ensg(gene_coords)
    gene_to_interval = build_gene_to_interval_map(gene_coords, intervals)

    ProbeIndex = list(range(8))

    clone_state_map = load_clone_state_map(args.state_labels_csv)
    print(f"[STATE MAP] {clone_state_map}")

    pb_expr = pd.read_csv(args.pseudo_bulk_expr, index_col=0)
    pb_prop = pd.read_csv(args.pseudo_bulk_prop, index_col=0)

    pb_expr_ensg, n_mapped_pb = map_symbols_to_ensg_columns(pb_expr, symbol_to_ensg)
    print(f"[PB] mapped SYMBOL->ENSG: {n_mapped_pb}/{pb_expr.shape[1]} | after collapse: {pb_expr_ensg.shape}")

    pb_interval = aggregate_genes_to_intervals_mean(pb_expr_ensg, gene_to_interval, n_intervals=n_intervals)
    TumorSample_all = pb_interval.T.to_numpy(dtype=float)

    FRefer_raw = pb_prop.to_numpy(dtype=float).T
    col_sums = FRefer_raw.sum(axis=0, keepdims=True)
    col_sums[col_sums < 1e-12] = 1.0
    FRefer_all = FRefer_raw / col_sums
    n_renorm = int((col_sums < 0.99).sum())
    print(f"[FREFER] Normalized {n_renorm} bulks with column sum <0.99")

    print("[CREF] Building W-weighted reference profiles from h5ad...")
    CRefer = build_cref_from_w_weighted_mean(
        annotated_h5ad=args.annotated_h5ad,
        target_genes=pd.Index(pb_expr.columns.astype(str)),
        symbol_to_ensg=symbol_to_ensg,
        gene_to_interval=gene_to_interval,
        n_intervals=n_intervals,
    )
    print(f"[CREF] shape: {CRefer.shape}")

    K_ref = CRefer.shape[1]
    K_prop = FRefer_all.shape[0]
    if K_ref != K_prop:
        raise ValueError(
            f"K mismatch: CRefer={K_ref} clones vs FRefer={K_prop} archetypes. "
            f"Check that W columns and proportion columns both have K={K_ref}."
        )

    rng = np.random.default_rng(args.seed)
    n_bulk_total = TumorSample_all.shape[1]
    need = args.n_cases * args.bulks_per_case
    if need > n_bulk_total:
        raise ValueError(f"Not enough pseudo-bulk samples: need {need}, have {n_bulk_total}")

    perm = rng.permutation(n_bulk_total)[:need]
    chunks = np.array_split(perm, args.n_cases)

    for i, idx in enumerate(chunks):
        TumorSample = TumorSample_all[:, idx]
        FRefer = FRefer_all[:, idx]
        out_pkl = out_dir / "BULK" / f"simulate_data_{i:03d}.pkl"
        write_mimo_pkl(
            out_pkl,
            TumorSample=TumorSample,
            CRefer=CRefer,
            FRefer=FRefer,
            ProbeIndex=ProbeIndex,
            clone_state_map=clone_state_map,
        )
        print(f"[WROTE] {out_pkl} | TumorSample={TumorSample.shape} CRefer={CRefer.shape} FRefer={FRefer.shape}")

    print("[DONE] All case files generated.")


if __name__ == "__main__":
    main()
