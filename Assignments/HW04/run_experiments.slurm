#!/bin/bash
#SBATCH --job-name=task3_experiments
#SBATCH --output=experiment_%j.out
#SBATCH --error=experiment_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8

SCHEDULING_TYPES=("static" "dynamic" "guided")
for sched in "${SCHEDULING_TYPES[@]}"; do
    echo "Running with $sched scheduling" > results_$sched.txt
    g++ -D$Sched task3.cpp -Wall -O3 -std=c++17 -o task3_$sched -fopenmp
    for threads in {1..8}; do
        ./task3_$sched 1000 100.0 $threads >> results_$sched.txt
    done
done