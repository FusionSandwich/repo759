#!/bin/bash
#SBATCH --job-name=task3_ts
#SBATCH --output=task3_ts_%j.out
#SBATCH --error=task3_ts_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=00:02:00
#SBATCH --partition=instruction

# Compile
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

n=1000000
t=8

echo "Time vs ts for n=$n, t=$t"
for ts_power in {1..10}; do
    ts=$(( 2**ts_power ))
    echo -n "ts=$ts => "
    srun ./task3 $n $t $ts
done
