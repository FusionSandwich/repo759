#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --output=task1_%j.out
#SBATCH --error=task1_%j.err
#SBATCH --time=00:00:50
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=instruction

module load nvidia/cuda/11.6.0

# Optional: check the nvcc version to ensure the module loaded correctly
#nvcc --version

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run the executable
./task1
