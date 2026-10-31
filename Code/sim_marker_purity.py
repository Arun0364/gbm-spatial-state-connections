import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/arun/Desktop/Research/Code")
warnings.filterwarnings("ignore")
from tree_compare import all_rooted_trees, case_name
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_improve_run import marker_matrix
from marker_weights import weights_marker

OUT = Path("/Users/arun/Desktop/Research/Results/marker_guided/sim_marker_purity.csv")
rows = []
for mix in (0.0, 0.75):
    for i, G in enumerate(all_rooted_trees(STATES, ROOT)):
        a, _, state = simulate(G, 500 + i, mix=mix)
        X, genes = marker_matrix(a)
        W = weights_marker(X, genes, json.loads(a.uns["state_markers"]), STATES, "gene_sd")
        Wt = a.obs[[f"W{k}" for k in range(len(STATES))]].values
        rows.append({"mix": mix, "case": case_name(G, ROOT), "seed": 500 + i,
                     "marker_purity": round(float(W.max(1).mean()), 4),
                     "true_purity": round(float(Wt.max(1).mean()), 4),
                     "n_spots": len(a), "cells_per_spot": round(float(a.obs["n_cells"].mean()), 2)})
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.groupby("mix")[["marker_purity", "true_purity", "n_spots", "cells_per_spot"]].agg(["mean", "min", "max"]).round(3).to_string())
