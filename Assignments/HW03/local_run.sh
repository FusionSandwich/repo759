#!/bin/bash
# local_run.sh
# If your shell is already set up for conda/mamba, you may comment out or adjust these lines.
# source "$(mamba info --root)/etc/profile.d/conda.sh"
# conda activate c_env

# Compile the code
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# Run the executable with example parameters (n = 1024, t = 4)
# and save the output to results_local.txt.
echo "Running task2 with n=1024 and t=4" > results_local.txt
./task2 1024 4 | tee -a results_local.txt
