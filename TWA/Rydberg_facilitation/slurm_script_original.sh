#!/bin/bash

#SBATCH -J Rydberg
#SBATCH -c 36
#SBATCH -t 50:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A_%a.err
#SBATCH -o out/%x_%A_%a.out
#SBATCH --mem 180G
#SBATCH --array=0-19  # 5 omega chunks × 4 trajectory chunks = 20 jobs

# Define chunks
OMEGA_RANGES=(
    "0 5" "5 10" "10 15" "15 20" "20 25"
)

TRAJ_RANGES=(
    "0 125" "125 250" "250 375" "375 500"
)

# Calculate omega and trajectory indices
OMEGA_IDX=$((SLURM_ARRAY_TASK_ID / 4))
TRAJ_IDX=$((SLURM_ARRAY_TASK_ID % 4))

# Get ranges
OMEGA_START=$(echo ${OMEGA_RANGES[$OMEGA_IDX]} | cut -d' ' -f1)
OMEGA_END=$(echo ${OMEGA_RANGES[$OMEGA_IDX]} | cut -d' ' -f2)
TRAJ_START=$(echo ${TRAJ_RANGES[$TRAJ_IDX]} | cut -d' ' -f1)
TRAJ_END=$(echo ${TRAJ_RANGES[$TRAJ_IDX]} | cut -d' ' -f2)

# Run Julia with both ranges
~/julia-1.11.2/bin/julia -t auto main.jl \
    --omega-start $OMEGA_START \
    --omega-end $OMEGA_END \
    --traj-start $TRAJ_START \
    --traj-end $TRAJ_END
