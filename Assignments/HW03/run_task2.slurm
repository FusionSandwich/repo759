#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --output=task2_%j.out
#SBATCH --error=task2_%j.err
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction


# Compile the program
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

# Define parameters and output file
n=1024
results_file="results.txt"
rm -f ${results_file}

# Run for t = 1 to 20
for t in {1..20}; do
    echo "Running with t=${t}" | tee -a ${results_file}
    output=$(./task2 ${n} ${t})
    echo "${output}" | tee -a ${results_file}
done
