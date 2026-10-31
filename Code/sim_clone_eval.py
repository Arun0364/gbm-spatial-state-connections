import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BASE    = Path("/Users/arun/Desktop/Research")
CASES   = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "Results/sim_clones/cases"
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "Results/sim_clones/clone_results.csv"


def spot_fractions(F, bulks, members, n_spots):
    acc = np.zeros((n_spots, F.shape[1]))
    hits = np.zeros(n_spots)
    for f, b in zip(F, bulks):
        acc[members[b]] += f
        hits[members[b]] += 1
    covered = hits > 0
    return acc[covered] / hits[covered, None], covered


def matched_r(A, B):
    k = A.shape[1]
    r = np.nan_to_num(np.corrcoef(A.T, B.T)[:k, k:])
    rows, cols = linear_sum_assignment(-r)
    return r[rows, cols]


def evaluate(d):
    t = np.load(d / "truth.npz")
    m = np.load(d / "mimo_F.npz")
    F, covered = spot_fractions(m["F"], m["bulks"], t["members"], len(t["Q"]))
    Q, W = t["Q"][covered], t["W_true"][covered]
    rq, rw, rqw = matched_r(F, Q), matched_r(F, W), matched_r(W, Q)
    case, seed = d.name.split("__s")
    return {"condition": d.parent.name, "case": case, "seed": int(seed), "spots": int(covered.sum()),
            "r_mimo_clone_mean": rq.mean(), "r_mimo_clone_min": rq.min(),
            "r_mimo_state_mean": rw.mean(), "r_state_clone_mean": rqw.mean(),
            "tracks": "clones" if rq.mean() > rw.mean() else "states"}


def main():
    rows = [evaluate(d) for d in sorted(CASES.glob("*/*")) if (d / "mimo_F.npz").exists()]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT_CSV, index=False)
    print(df.groupby("condition")[["r_mimo_clone_mean", "r_mimo_clone_min", "r_mimo_state_mean", "r_state_clone_mean"]].mean().round(3))
    print(df.groupby("condition")["tracks"].value_counts())


if __name__ == "__main__":
    main()
