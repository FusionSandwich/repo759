#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --output=task2_%A_%a.out
#SBATCH --error=task2_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

# Activate the mamba environment "c_env" if needed:
# source "$(mamba info --root)/etc/profile.d/conda.sh"
# conda activate c_env

# Compile the code
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# Define parameters and output file for the collected results
n=1024
results_file="results.txt"
rm -f ${results_file}

# Run the executable for thread counts from 1 to 20 and save the results.
for t in {1..20}; do
    echo "Running with t=${t}" | tee -a ${results_file}
    # Capture the output from the run (first element, last element, and runtime)
    output=$(./task2 ${n} ${t})
    echo "${output}" | tee -a ${results_file}
done

# Generate a plot from the results using the Python script below (optional)
python3 plot_task2.py ${results_file} task2.pdf
