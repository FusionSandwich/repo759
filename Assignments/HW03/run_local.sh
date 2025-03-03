#!/bin/bash
# run_local_save.sh
# This script compiles your code with OpenMP, runs it for thread counts 1 to 20,
# and saves the output to run_local_output.txt while still printing to the terminal.

# Define output file
OUTPUT_FILE="run_local_output.txt"

# Remove previous output file if it exists
rm -f "$OUTPUT_FILE"

# Check if the conda initialization file exists and source it if so.
CONDA_INIT="$HOME/mambaforge/etc/profile.d/conda.sh"
if [ -f "$CONDA_INIT" ]; then
    source "$CONDA_INIT"
    conda activate myenv
else
    echo "Warning: Conda initialization script not found at $CONDA_INIT." | tee -a "$OUTPUT_FILE"
    echo "If you need to activate your environment, run 'conda init' or activate manually." | tee -a "$OUTPUT_FILE"
fi

# Compile the code using g++
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -fopenmp -o task1 | tee -a "$OUTPUT_FILE"
if [ $? -ne 0 ]; then
    echo "Compilation failed." | tee -a "$OUTPUT_FILE"
    exit 1
fi
echo "Compilation successful." | tee -a "$OUTPUT_FILE"

# Run the executable for n = 1024 and t = 1 to 20
for t in {1..20}; do
    echo "-----------------------------------------" | tee -a "$OUTPUT_FILE"
    echo "Running with t=$t" | tee -a "$OUTPUT_FILE"
    ./task1 1024 $t | tee -a "$OUTPUT_FILE"
done

# Deactivate conda environment if conda is available
if command -v conda &> /dev/null; then
    conda deactivate
fi
