#!/bin/bash
#SBATCH --job-name=plot_task3
#SBATCH --output=plot_task3_%j.out
#SBATCH --error=plot_task3_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction

module load nvidia/cuda

./run_task1.sh