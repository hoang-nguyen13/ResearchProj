#!/bin/bash

#SBATCH -J Rydberg
#SBATCH -c 36
#SBATCH -t 50:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem 180G
#SBATCH --array=0-1  # 2 chunks for 2 nodes
#SBATCH --nodes=1    # Request 1 node per array task

# Define omega chunks (split into 2)
OMEGA_RANGES=(
    "0 12.5"    # First half for node 1
    "12.5 25"   # Second half for node 2
)


# Get omega range
OMEGA_START=$(echo ${OMEGA_RANGES[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f1)
OMEGA_END=$(echo ${OMEGA_RANGES[$SLURM_ARRAY_TASK_ID]} | cut -d' ' -f2)

# Run Julia with omega range only
~/julia-1.11.2/bin/julia -t auto main.jl \
    --omega-start $OMEGA_START \
    --omega-end $OMEGA_END
