#!/bin/bash
#SBATCH --job-name=CheckDir
#SBATCH --partition=instruction
#SBATCH --output=CheckDir.out
#SBATCH --error=CheckDir.err
#SBATCH --time=0-00:01:00
# Print the current working directory
echo "Job started in directory: $(pwd)"