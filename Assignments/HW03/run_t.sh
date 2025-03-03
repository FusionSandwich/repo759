#!/bin/bash
#SBATCH --job-name=task3_t
#SBATCH --output=task3_t_%j.out
#SBATCH --error=task3_t_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=instructional

module load gcc/9.3.0

# Compile
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

n=1000000
best_ts=32

echo "Time vs t for n=$n, ts=$best_ts"
for t in {1..20}; do
    echo -n "t=$t => "
    srun ./task3 $n $t $best_ts
done
