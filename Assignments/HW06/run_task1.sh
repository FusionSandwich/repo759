#!/bin/bash

# Load the CUDA module
module load nvidia/cuda
# Check if module load was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to load nvidia/cuda module."
    exit 1
fi

# Define the threads per block values
TPB1=1024
TPB2=512 # Choose your second value here

# Output data files
DATA_FILE1="task1_times_${TPB1}.dat"
DATA_FILE2="task1_times_${TPB2}.dat"
PLOT_FILE="task1.pdf"

# Clean previous data files
rm -f $DATA_FILE1 $DATA_FILE2 $PLOT_FILE task1

echo "Compiling task1..."
# Compile task1.cu and matmul.cu [cite: 15]
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi
echo "Compilation successful."

echo "Running experiments for threads_per_block = $TPB1..."
echo "# N   Time(ms)" > $DATA_FILE1 # Add header
# Loop through n = 2^5 to 2^14 [cite: 16]
for (( p=5; p<=14; p++ )); do
    n=$((2**p))
    echo "Running with n = $n, threads = $TPB1"
    # Run the executable and capture the second line of output (time)
    output=$(./task1 $n $TPB1)
    if [ $? -ne 0 ]; then
        echo "Error: Execution failed for n=$n, threads=$TPB1"
        continue # Skip this data point
    fi
    # Extract the second line (timing)
    time_ms=$(echo "$output" | sed -n '2p')
    echo "$n $time_ms" >> $DATA_FILE1
    echo "  Time: $time_ms ms"
done

echo "Running experiments for threads_per_block = $TPB2..."
echo "# N   Time(ms)" > $DATA_FILE2 # Add header
# Loop through n = 2^5 to 2^14 [cite: 16]
for (( p=5; p<=14; p++ )); do
    n=$((2**p))
    echo "Running with n = $n, threads = $TPB2"
    # Run the executable and capture the second line of output (time)
    output=$(./task1 $n $TPB2)
    if [ $? -ne 0 ]; then
        echo "Error: Execution failed for n=$n, threads=$TPB2"
        continue # Skip this data point
    fi
     # Extract the second line (timing)
    time_ms=$(echo "$output" | sed -n '2p')
    echo "$n $time_ms" >> $DATA_FILE2
     echo "  Time: $time_ms ms"
done

echo "Data collection complete."
echo "Data saved to $DATA_FILE1 and $DATA_FILE2"

# Optional: Call python script to plot if python/matplotlib available
# echo "Generating plot..."
# module load python # Load python if needed
# python plot_task1.py
# if [ $? -ne 0 ]; then
#    echo "Error: Plot generation failed."
# else
#    echo "Plot saved as $PLOT_FILE"
# fi

exit 0