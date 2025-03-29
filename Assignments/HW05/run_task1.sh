#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --output=task1_%j.out
#SBATCH --error=task1_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --partition=instruction

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run the executable
./task1
