#!/bin/bash
# Script to run task1 for n = 2^5 to 2^14 with two threads_per_block values

echo "n time_1024" > times_1024.txt
for i in {5..14}; do
    n=$((1 << i))  # 2^i
    time=$(./task1 $n 1024 | tail -n 1)
    echo "$n $time" >> times_1024.txt
done

echo "n time_256" > times_256.txt
for i in {5..14}; do
    n=$((1 << i))  # 2^i
    time=$(./task1 $n 256 | tail -n 1)
    echo "$n $time" >> times_256.txt
done