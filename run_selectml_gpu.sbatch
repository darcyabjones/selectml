#!/bin/bash --login

#SBATCH --partition=gpuq
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=y95
#SBATCH --export=NONE

# TOPAZ doesn't play nice with conda without this
unset PYTHONPATH

#Default loaded compiler module is gcc module
module unload cuda
module unload gcc

module load gcc/11.1.0
module load cuda/11.4.2

source ~/.bashrc
conda activate ${PWD}/condaenv
LD_LIBRARY_PATH="${PWD}/condaenv/lib/:${LD_LIBRARY_PATH:-}"

set -euo pipefail

TASK=$1
shift
DATASET=$1
shift

MODEL=tf

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
  ${TASK} \
  ${MODEL} \
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
  --ntasks 1
  --ntrials 1000 \
  --timeout 23
