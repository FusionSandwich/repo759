#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction

module load nvidia/cuda/11.6.0

# Compile task2.cu into an executable named "task2"
nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Run the executable
./task2
