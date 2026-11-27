set -e
cd "$(dirname "$0")"
V="$(cd .. && pwd)/.venv/bin/python"
P13=${MIMO_PYTHON:-python3.13}
R="$(cd .. && pwd)/Results/sim_clones"
$V sim_clone_cases.py --seeds 5 --workers 6 --out $R/cases
ls -d $R/cases/*/* | xargs -P 4 -n 4 $P13 mimo_solve.py
$V sim_clone_eval.py $R/cases $R/clone_results.csv
echo "CLONES DONE"
