#!/bin/bash --login

#SBATCH --partition=longq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --account=y95
#SBATCH --export=NONE

module load nextflow
module load java

srun --export=all nextflow run -c ./zeus.config -resume ./main.nf 
