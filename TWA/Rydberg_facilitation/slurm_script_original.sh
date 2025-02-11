#!/bin/bash

#SBATCH -J Rydberg
#SBATCH -c 36
#SBATCH -t 50:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem 180G
#SBATCH --array=0-4

# Define Omega ranges for 5 chunks (adjust based on your needs)
OMEGA_RANGES=(
    "0 5"
    "5 10"
    "10 15"
    "15 20"
    "20 25"
)

# Get the range for this array task
OMEGA_START=$(echo ${OMEGA_RANGES[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f1)
OMEGA_END=$(echo ${OMEGA_RANGES[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f2)

# Run Julia with the specific Omega range
~/julia-1.11.2/bin/julia -t auto main.jl --omega-start $OMEGA_START --omega-end $OMEGA_END
