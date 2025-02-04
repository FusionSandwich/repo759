#!/bin/bash
#SBATCH --job-name=task6
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output=task6_%j.out
#SBATCH --error=task6_%j.err

# Compile the program (ensure it's up-to-date)
g++ task6.cpp -Wall -O3 -std=c++17 -o task6

srun ./task6 "$1"
