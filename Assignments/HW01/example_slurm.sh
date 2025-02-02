#!/usr/bin/env zsh
#SBATCH --partition=research
#SBATCH --time 00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=example.output
# log in the submission directory
cd $SLURM_SUBMIT_DIR
# load the gcc for compiling C++ programs
module load gcc/13.2.0
# load the nvcc for compiling cuda programs
module load nvidia/cuda
# clone (replace the github link to yours)
git clone https://github.com/FusionSandwich/repo759.git
cd repo759/HW01
g++ example.cpp -o example
./example