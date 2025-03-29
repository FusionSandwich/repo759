#!/bin/bash
#SBATCH --job-name=plot_task3
#SBATCH --output=plot_task3_%j.out
#SBATCH --error=plot_task3_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=instruction
#SBATCH --gpus-per-task=1

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run the Python script to collect timings and generate the plot.
python3 plot_task3.py
