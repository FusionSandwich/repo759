#!/bin/bash
#SBATCH --job-name=task3_ts
#SBATCH --output=task3_ts_%j.out
#SBATCH --error=task3_ts_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=instructional

module load gcc/9.3.0   # or whichever compiler / modules you need

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
