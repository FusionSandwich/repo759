#!/bin/bash
#SBATCH --partition=instruction
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=reduce_benchmark
#SBATCH --output=reduce_benchmark.out
#SBATCH --error=hw07_task2_%j.err

# Load CUDA module
module load nvidia/cuda/11.8

# Compile the program
nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Create output files for the timing data
echo "N,time_1024,time_256" > timing_data.csv

# Run the program for different values of N
# Start with smaller values and increase gradually
for i in {10..20}
do
    N=$((2**i))
    echo "Running with N = $N"
    
    # Run with threads_per_block = 1024
    echo "Running with threads_per_block = 1024"
    ./task2 $N 1024 > temp_output_1024.txt 2>&1
    if [ $? -eq 0 ]; then
        sum_1024=$(head -n 1 temp_output_1024.txt)
        time_1024=$(tail -n 1 temp_output_1024.txt)
        echo "Sum: $sum_1024, Time: $time_1024 ms"
    else
        echo "Error occurred with N=$N, threads_per_block=1024"
        cat temp_output_1024.txt
        time_1024="NA"
    fi
    
    # Run with threads_per_block = 256
    echo "Running with threads_per_block = 256"
    ./task2 $N 256 > temp_output_256.txt 2>&1
    if [ $? -eq 0 ]; then
        sum_256=$(head -n 1 temp_output_256.txt)
        time_256=$(tail -n 1 temp_output_256.txt)
        echo "Sum: $sum_256, Time: $time_256 ms"
    else
        echo "Error occurred with N=$N, threads_per_block=256"
        cat temp_output_256.txt
        time_256="NA"
    fi
    
    # Save the results
    echo "$N,$time_1024,$time_256" >> timing_data.csv
done

# Run larger values with more caution, skipping some intermediate values to save time
for i in {22..30..2}
do
    N=$((2**i))
    echo "Running with N = $N"
    
    # Check if we can allocate memory for this size
    max_mem=$((N * 4 * 2)) # Rough estimate of max memory needed in bytes
    max_mem_gb=$(echo "scale=2; $max_mem / 1073741824" | bc)
    echo "Estimated max memory: $max_mem_gb GB"
    
    # Skip if estimated memory is too large
    if (( $(echo "$max_mem_gb > 15" | bc -l) )); then
        echo "Skipping N=$N due to potential memory constraints"
        echo "$N,NA,NA" >> timing_data.csv
        continue
    fi
    
    # Run with threads_per_block = 1024
    echo "Running with threads_per_block = 1024"
    ./task2 $N 1024 > temp_output_1024.txt
    sum_1024=$(head -n 1 temp_output_1024.txt)
    time_1024=$(tail -n 1 temp_output_1024.txt)
    echo "Sum: $sum_1024, Time: $time_1024 ms"
    
    # Run with threads_per_block = 256
    echo "Running with threads_per_block = 256"
    ./task2 $N 256 > temp_output_256.txt
    sum_256=$(head -n 1 temp_output_256.txt)
    time_256=$(tail -n 1 temp_output_256.txt)
    echo "Sum: $sum_256, Time: $time_256 ms"
    
    # Save the results
    echo "$N,$time_1024,$time_256" >> timing_data.csv
done
