#!/bin/bash --login

#SBATCH --partition=longq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --account=y95
#SBATCH --export=NONE

module load nextflow
module load java

srun --export=all nextflow run -c ./zeus.config -with-conda /scratch/y95/darcyabjones/condaenv/ -resume ./main.nf --infiles "/scratch/y95/darcyabjones/run/A{1,100,1000,10000}_N{500,1000,5000,10000}_M{1000,5000,10000}*"  --trials 300 --task regression
