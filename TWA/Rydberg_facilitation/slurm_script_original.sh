#!/bin/bash

#SBATCH -J Rydberg
#SBATCH -c 36
#SBATCH -t 50:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem 180G
#SBATCH --array=0-50  # 51 values from 0 to 25 in steps of 0.5

# Single omega values array
OMEGA_VALUES=(
    0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 
    5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 
    10.5 11 11.5 12 12.5 13 13.5 14 14.5 
    15 15.5 16 16.5 17 17.5 18 18.5 19 19.5 
    20 20.5 21 21.5 22 22.5 23 23.5 24 24.5 25
)

# Get single omega value
OMEGA=${OMEGA_VALUES[$SLURM_ARRAY_TASK_ID]}

# Run Julia with single omega value
~/julia-1.11.2/bin/julia -t auto main.jl --omega-start $OMEGA
