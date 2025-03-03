#!/usr/bin/env bash

# Compile the code
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Experiment 1: Vary threshold (ts) with fixed thread count (t=8) and n=1e6.
n=1000000
t=8
echo "Time vs. threshold (n=$n, t=$t):"
for threshold in 2 4 8 16 32 64 128 256 512 1024; do
    echo -n "threshold=$threshold => "
    ./task3 $n $t $threshold
done

# Experiment 2: Vary number of threads (t) with fixed threshold (e.g., threshold=32) and n=1e6.
best_threshold=32
echo "Time vs. threads (n=$n, threshold=$best_threshold):"
for threads in {1..20}; do
    echo -n "t=$threads => "
    ./task3 $n $threads $best_threshold
done
