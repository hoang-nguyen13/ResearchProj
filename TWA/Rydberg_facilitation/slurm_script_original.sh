#!/bin/bash

#SBATCH -J Rydberg
#SBATCH -c 36
#SBATCH -t 50:00:00
#SBATCH -p physik-fleischhauer
#SBATCH -e err/%x_%A.err
#SBATCH -o out/%x_%A.out
#SBATCH --mem 180G
 
~/julia-1.11.2/bin/julia -t auto main.jl


