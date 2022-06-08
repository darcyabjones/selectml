#!/bin/bash --login

#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH --account=y95
#SBATCH --export=NONE

conda activate ${PWD}/condaenv

set -euo pipefail

TASK=$1
MODEL=$2
DATASET=$3


if [ -s "${DATASET}/${TASK}_${MODEL}_pickle.pkl" ]
then
    cp "${DATASET}/${TASK}_${MODEL}_pickle.pkl" "${DATASET}/${TASK}_${MODEL}_continue.pkl"
    CONTINUE="--continue ${DATASET}/${TASK}_${MODEL}_continue.pkl"
else
    CONTINUE=""
fi

selectml \
  optimise \
  regression \
  bglr \
  "${DATASET}/markers_train.tsv" \
  "${DATASET}/phenos_train.tsv" \
  -r response \
  -n name \
  -g l_{01..11} \
  -o "${DATASET}/${TASK}_${MODEL}_results.tsv" \
  --full "${DATASET}/${TASK}_${MODEL}_full_results.tsv" \
  --pickle "${DATASET}/${TASK}_${MODEL}_pickle.pkl" \
  ${CONTINUE} \
  --importance "${DATASET}/${TASK}_${MODEL}_importance.tsv" \
  --best "${DATASET}/${TASK}_${MODEL}_best.json" \
  --ntasks 1
  --cpu 1 \
  --ntrials 1000 \
  --timeout 23
