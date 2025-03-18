#!/bin/bash
#SBATCH --job-name=task4
#SBATCH --output=experiment_%j.out
#SBATCH --error=experiment_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=instruction

# Compile the code once
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

SCHEDULING_TYPES=("static" "dynamic" "guided")
for sched in "${SCHEDULING_TYPES[@]}"; do
    echo "Running with $sched scheduling" > results_$sched.txt
    for threads in {1..8}; do
        ./task3 1000 100.0 $threads $sched >> results_$sched.txt
    done
done
