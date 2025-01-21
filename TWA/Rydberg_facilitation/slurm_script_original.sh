#!/bin/bash

#SBATCH --array=1-1
#SBATCH -J Waveguide
#SBATCH -c 16
#SBATCH -t 50:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A.err
#SBATCH -o out/%x_%A.out
#SBATCH --mem 240G


# Reihenfolge Paramter: Omega nAtoms 
~/julia-1.11.2/bin/julia -t auto waveguide.jl $SLURM_ARRAY_TASK_ID


