#!/bin/bash --login

#SBATCH --partition=debugq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=1:00:00
#SBATCH --account=y95
#SBATCH --export=NONE

source ~/.bashrc
conda activate ${PWD}/condaenv

set -euo pipefail

TASK=$1
shift
MODEL=$1
shift
DATASET=$1
shift

if [ -s "${DATASET}/${TASK}_${MODEL}_pickle.pkl" ]
then
    cp "${DATASET}/${TASK}_${MODEL}_pickle.pkl" "${DATASET}/${TASK}_${MODEL}_continue.pkl"
    CONTINUE="--continue ${DATASET}/${TASK}_${MODEL}_continue.pkl"
else
    CONTINUE=""
fi

if [ $# -gt 0 ]
then
    GROUPING="-g ${@}"
else
    GROUPING=""
fi

selectml \
  optimise \
  "${TASK}" \
  "${MODEL}" \
  "${DATASET}/markers_train.tsv" \
  "${DATASET}/phenos_train.tsv" \
  -r response \
  -n name \
  ${GROUPING} \
  -o "${DATASET}/${TASK}_${MODEL}_results.tsv" \
  --full "${DATASET}/${TASK}_${MODEL}_full_results.tsv" \
  --pickle "${DATASET}/${TASK}_${MODEL}_pickle.pkl" \
  ${CONTINUE} \
  --importance "${DATASET}/${TASK}_${MODEL}_importance.tsv" \
  --best "${DATASET}/${TASK}_${MODEL}_best.json" \
  --ntasks 1 \
  --ntrials 1000 \
  --timeout 23
