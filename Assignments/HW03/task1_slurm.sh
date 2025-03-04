#!/bin/bash
#SBATCH --job-name=CheckDir
#SBATCH --partition=instruction
#SBATCH --output=CheckDir.out
#SBATCH --error=CheckDir.err
#SBATCH --time=0-00:01:00
#SBATCH --cpus-per-task=20

# Compile
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -fopenmp -o task1

# Run for n=1024, t=1..20
for t in $(seq 1 20); do
    echo "Running with t=$t"
    ./task1 1024 $t
done
