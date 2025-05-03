#!/bin/bash
#SBATCH --partition=instruction      # run in the instructional GPU partition
#SBATCH --job-name=hw07_task1
#SBATCH --output=hw07_task1_%j.out
#SBATCH --error=hw07_task1_%j.err
#SBATCH --time=00:50:00               # hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:2                  # request one GPU

# 1) Load CUDA toolkit
module load nvidia/cuda/11.8

# 2) Compile your code
nvcc task1.cu matmul.cu \
     -Xcompiler -O3 -Xcompiler -Wall \
     -Xptxas -O3 \
     -std=c++17 \
     -o task1

# ----- PART (b) -----
# run for n = 2^5,2^6,…,2^14 with a fixed block_dim
BLOCK_DIM=16                             # pick any—as a first pass—for part (b)
OUT_N=task1_times.txt
echo "# n time_int time_float time_double" > ${OUT_N}
for EXP in {5..14}; do
    N=$((1<<EXP))
    # task1 prints: first_elem\nlast_elem\ntime_ms\n (three times, one per data type)
    RESULT=$(./task1 ${N} ${BLOCK_DIM})
    INT_TIME=$(echo "$RESULT" | awk 'NR==3')
    FLOAT_TIME=$(echo "$RESULT" | awk 'NR==6')
    DOUBLE_TIME=$(echo "$RESULT" | awk 'NR==9')
    echo "${N} ${INT_TIME} ${FLOAT_TIME} ${DOUBLE_TIME}" >> ${OUT_N}
done
echo "Part (b) complete → ${OUT_N}"

# ----- PART (c) -----
# sweep BLOCK_DIM for n = 2^14 to find the fastest one
N=16384
OUT_BD=blockdim_sweep.txt
echo "# block_dim time_int time_float time_double" > ${OUT_BD}
for BD in 4 8 16 24 32; do  # Try more block dimensions for better analysis
    RESULT=$(./task1 ${N} ${BD})
    INT_TIME=$(echo "$RESULT" | awk 'NR==3')
    FLOAT_TIME=$(echo "$RESULT" | awk 'NR==6')
    DOUBLE_TIME=$(echo "$RESULT" | awk 'NR==9')
    echo "${BD} ${INT_TIME} ${FLOAT_TIME} ${DOUBLE_TIME}" >> ${OUT_BD}
done
echo "Part (c) complete → ${OUT_BD}"