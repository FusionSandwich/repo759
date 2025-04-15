#!/bin/bash
#SBATCH --job-name=plot_task1 # Adjusted name slightly to match task
#SBATCH --output=task1_results_%j.out # Output file for results
#SBATCH --error=task1_errors_%j.err  # Error file
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction
#SBATCH --mem=4G # Request some memory, adjust if needed

# Load necessary modules [cite: 5]
module load nvidia/cuda/11.8

# Compile the code [cite: 15]
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Check if compilation was successful
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

# Define threads per block values
THREADS_1=1024
THREADS_2=512 # Choose another value, e.g., 512 [cite: 16]

# Output file names based on threads_per_block
OUTPUT_FILE_1="results_t${THREADS_1}.txt"
OUTPUT_FILE_2="results_t${THREADS_2}.txt"

# Clear previous results files
> $OUTPUT_FILE_1
> $OUTPUT_FILE_2

echo "Running with threads_per_block = ${THREADS_1}"
echo "N Time(ms)" > $OUTPUT_FILE_1 # Add header
# Loop through matrix sizes n=2^5 to 2^14 [cite: 16]
for i in {5..14}; do
  n=$((2**i))
  echo "Running task1 for n = $n with ${THREADS_1} threads..."
  # Run the executable and capture the time (second line of output)
  output=$(./task1 $n $THREADS_1)
  time_ms=$(echo "$output" | tail -n 1)
  echo "$n $time_ms" >> $OUTPUT_FILE_1
  echo "Last element: $(echo "$output" | head -n 1), Time: ${time_ms} ms"
done

echo "Running with threads_per_block = ${THREADS_2}"
echo "N Time(ms)" > $OUTPUT_FILE_2 # Add header
# Loop through matrix sizes n=2^5 to 2^14
for i in {5..14}; do
  n=$((2**i))
  echo "Running task1 for n = $n with ${THREADS_2} threads..."
   # Run the executable and capture the time (second line of output)
  output=$(./task1 $n $THREADS_2)
  time_ms=$(echo "$output" | tail -n 1)
  echo "$n $time_ms" >> $OUTPUT_FILE_2
   echo "Last element: $(echo "$output" | head -n 1), Time: ${time_ms} ms"
done

echo "Job finished. Results saved in ${OUTPUT_FILE_1} and ${OUTPUT_FILE_2}"

# The plotting script plot_task1.py should be run locally
# after transferring the results_t*.txt files.