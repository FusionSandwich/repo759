# Load the CUDA module.
module load nvidia/cuda/11.8

# Compile the stencil code.
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Define parameters for the experiments.
R=128
THREADS1=1024   # First case: threads per block = 1024
THREADS2=512    # Second case: alternative threads per block value
START_EXP=10    # Starting exponent for n (2^10)
END_EXP=29      # Ending exponent for n (2^29)

# Remove any previous data files.
rm -f data_1024.txt data_512.txt

# Add headers to data files for clarity (optional but recommended).
echo "n time" > data_1024.txt
echo "n time" > data_512.txt

# === Warm-up Run for THREADS1 ===
echo "Warming up for threads per block = ${THREADS1}"
./task2 $((1 << START_EXP)) $R $THREADS1 > /dev/null

echo "Running experiments for threads per block = ${THREADS1}"
# Loop over exponents for threads per block = 1024.
for exponent in $(seq $START_EXP $END_EXP); do
    n=$((1 << exponent))  # Calculate n = 2^exponent
    output=$(./task2 $n $R $THREADS1)
    exec_time=$(echo "$output" | sed -n '2p')
    echo "$n $exec_time" >> data_1024.txt
done

# === Warm-up Run for THREADS2 ===
echo "Warming up for threads per block = ${THREADS2}"
./task2 $((1 << START_EXP)) $R $THREADS2 > /dev/null

echo "Running experiments for threads per block = ${THREADS2}"
# Loop over exponents for threads per block = 512.
for exponent in $(seq $START_EXP $END_EXP); do
    n=$((1 << exponent))  # Calculate n = 2^exponent
    output=$(./task2 $n $R $THREADS2)
    exec_time=$(echo "$output" | sed -n '2p')
    echo "$n $exec_time" >> data_512.txt
done

echo "Data collection complete. Data files (data_1024.txt and data_512.txt) have been generated."
echo "Please transfer these data files to a system with Python and run 'python3 plot_task2.py' to generate the plot."
