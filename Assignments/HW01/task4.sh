#!/bin/bash
#SBATCH --job-name=FirstSlurm
#SBATCH --partition=instruction
#SBATCH --cpus-per-task=2
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err

# Print the hostname of the compute node
hostname