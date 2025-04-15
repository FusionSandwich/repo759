#!/bin/bash
#SBATCH --job-name=run_plot_task2      # Adjusted job name for task2
#SBATCH --output=run_plottask2%j.out    # Adjusted output file
#SBATCH --error=run_plottask2%j.err     # Adjusted error file
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction

# Load the CUDA module.
module load nvidia/cuda/11.8

# Compile the stencil code.
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Define parameters for the experiments.
R=128
THREADS1=1024   # First case: threads per block = 1024
THREADS2=512    # Second case: alternative threads per block value (choose any valid value)
START_N=210
END_N=229

# Remove any previous data files.
rm -f data_1024.txt data_512.txt

echo "Running experiments for threads per block = ${THREADS1}"
# Loop over n for threads per block = 1024.
for n in $(seq $START_N $END_N); do
    # Execute task2 with given n, R, THREADS1.
    output=$(./task2 $n $R $THREADS1)
    # task2 prints two lines:
    #   Line 1: last element of the output array (not used here).
    #   Line 2: kernel execution time (in milliseconds).
    exec_time=$(echo "$output" | sed -n '2p')
    echo "$n $exec_time" >> data_1024.txt
done

echo "Running experiments for threads per block = ${THREADS2}"
# Loop over n for threads per block = 512.
for n in $(seq $START_N $END_N); do
    output=$(./task2 $n $R $THREADS2)
    exec_time=$(echo "$output" | sed -n '2p')
    echo "$n $exec_time" >> data_512.txt
done

echo "Data collection complete. Data files (data_1024.txt and data_512.txt) have been generated."
echo "Please transfer these data files to a system with Python and run 'python3 plot_task2.py' to generate the plot."
