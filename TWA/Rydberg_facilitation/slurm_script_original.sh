#!/bin/bash

#SBATCH -J Rydberg
#SBATCH -c 16
#SBATCH -t 50:00:00
#SBATCH -p epyc-256
#SBATCH -e err/%x_%A.err
#SBATCH -o out/%x_%A.out
#SBATCH --mem 240G
 
~/julia-1.11.2/bin/julia -t auto main.jl


