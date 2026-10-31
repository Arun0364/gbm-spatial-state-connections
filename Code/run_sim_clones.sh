set -e
cd "$(dirname "$0")"
V=/Users/arun/Desktop/Research/.venv/bin/python
P13=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
R=/Users/arun/Desktop/Research/Results/sim_clones
$V sim_clone_cases.py --seeds 5 --workers 6 --out $R/cases
ls -d $R/cases/*/* | xargs -P 4 -n 4 $P13 mimo_solve.py
$V sim_clone_eval.py $R/cases $R/clone_results.csv
echo "CLONES DONE"
