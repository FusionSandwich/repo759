#!/bin/bash
#SBATCH --job-name=plot_task3
#SBATCH --output=plot_task3_%j.out
#SBATCH --error=plot_task3_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=instruction

module load nvidia/cuda/11.6.0
module load conda/miniforge/23.1.0

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

./task3
