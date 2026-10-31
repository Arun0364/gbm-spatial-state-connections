import sys
import argparse
from pathlib import Path
import numpy as np

MIMO_DIR = Path("/Users/arun/Desktop/Research/FISH_deconvolution/MIMO/MIMOsolver")
sys.path.insert(0, str(MIMO_DIR))
import MIMOsolver as gs


def solve_case(path, alphaf=1.0, alpha1=1.0, alpha2=1.0, max_steps=10, tol=1e-4):
    d = np.load(path)
    bulk = d["TumorSample"].T
    sample_cells = d["CRefer"].T
    clone_freq = d["FRefer"].T
    k = sample_cells.shape[0]
    probes = list(range(8))
    fish = np.full((8, k), 2.0).T
    pr = np.diag([2.0] * k)
    ctotal = np.zeros((2 * k, sample_cells.shape[1]))
    ctotal[k:] = sample_cells
    ctotal = np.concatenate([ctotal, 2 * np.ones((1, sample_cells.shape[1]))])
    mask = np.arange(bulk.shape[1])
    old = 0.0
    for step in range(1, max_steps + 1):
        F, _ = gs.updateProportion(bulk, ctotal, pr, clone_freq, alphaf, k, root=2 * k)
        S1, _ = gs.updateTree(ctotal, k, alpha1, root=2 * k)
        C, obj = gs.updateCopyNum(bulk, F, S1, None, sample_cells, pr, k, fish, probes, mask=mask,
                                  alpha1=alpha1, alpha2=alpha2, Cap=10)
        fish_inf = C[:, probes]
        pr, _ = gs.updatePloidy(bulk, C, F, fish, fish_inf, np.diag([2.0] * k), alpha2)
        ctotal[:k] = C
        if abs(old - obj) < tol:
            break
        old = obj
    return np.asarray(F), np.asarray(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dirs", nargs="+")
    args = ap.parse_args()
    for cd in args.case_dirs:
        cd = Path(cd)
        if (cd / "mimo_F.npz").exists():
            continue
        Fs, idx = [], []
        for case in sorted(cd.glob("case_*.npz")):
            F, _ = solve_case(case)
            Fs.append(F)
            idx.append(np.load(case)["bulks"])
        np.savez(cd / "mimo_F.npz", F=np.concatenate(Fs, axis=0), bulks=np.concatenate(idx))
        print("solved", cd.name, flush=True)


if __name__ == "__main__":
    main()
