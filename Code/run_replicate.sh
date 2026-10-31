#!/bin/zsh
cd /Users/arun/Desktop/Research
MODE=$1
RUNS=(
  GSE242352:SPA1_D GSE242352:SPA2_A GSE242352:SPA3_D GSE242352:SPA5_D GSE242352:SPA7_D GSE242352:SPA8_A
  GSE253080:GBM1-R1-1 GSE253080:GBM1-R1-2 GSE253080:GBM1-R1-3 GSE253080:GBM1-R1-4
  GSE253080:GBM1-R2-1 GSE253080:GBM1-R2-2 GSE253080:GBM1-R2-3 GSE253080:GBM1-R2-4
  GSE253080:GBM2-R1 GSE253080:GBM2-R2 GSE253080:GBM2-R3 GSE253080:GBM2-R4
  GSE253080:GBM3-R1 GSE253080:GBM3-R2 GSE253080:GBM3-R3 GSE253080:GBM3-R4
)
mkdir -p Results/replicate_logs
for r in $RUNS; do
  ds=${r%%:*}
  sec=${r##*:}
  log=Results/replicate_logs/${sec}_${MODE}.log
  if [[ $MODE == pre ]]; then
    if [[ -f Results/$ds/$sec/${sec}_norm_counts_tumor.mat ]]; then echo "skip $sec (already prepared)"; continue; fi
    echo ">>> $sec prepare / annotate / export"
    { .venv/bin/python Code/replicate_prepare.py $ds $sec &&
      Rscript Code/replicate_annotate.R Results/$ds/$sec $sec &&
      .venv/bin/python Code/replicate_classify_export.py $ds $sec } > $log 2>&1 || { echo "FAILED $sec, see $log"; continue; }
    grep -E "after QC|UNMIX input" $log
  elif [[ $MODE == analyze ]]; then
    if [[ ! -f Results/$ds/$sec/${sec}_tumor_unmix_results.mat ]]; then echo "skip $sec (no MATLAB result)"; continue; fi
    echo ">>> $sec analyze"
    .venv/bin/python Code/replicate_analyze.py $ds $sec > $log 2>&1 || { echo "FAILED $sec, see $log"; continue; }
    grep -E "tumor spots;" $log
  else
    echo "usage: zsh Code/run_replicate.sh pre|analyze"
    exit 1
  fi
done
echo "Done ($MODE)."