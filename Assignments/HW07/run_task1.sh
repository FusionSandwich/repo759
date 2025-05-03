#!/bin/bash
#SBATCH --partition=instructional      # run in the instructional GPU partition
#SBATCH --job-name=hw07_task1
#SBATCH --output=hw07_task1_%j.out
#SBATCH --error=hw07_task1_%j.err
#SBATCH --time=01:00:00               # hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1                  # request one GPU

# 1) Load CUDA toolkit
module load nvidia/cuda

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
echo "# n time_ms" > ${OUT_N}
for EXP in {5..14}; do
    N=$((1<<EXP))
    # task1 prints: first_elem\nlast_elem\ntime_ms\n
    T=$(./task1 ${N} ${BLOCK_DIM} | tail -n1)
    echo "${N} ${T}" >> ${OUT_N}
done
echo "Part (b) complete → ${OUT_N}"

# ----- PART (c) -----
# sweep BLOCK_DIM for n = 2^14 to find the fastest one
N=16384
OUT_BD=blockdim_sweep.txt
echo "# block_dim time_ms" > ${OUT_BD}
for BD in 8 16 32; do         # you can extend this list: 4,8,16,32…
    T=$(./task1 ${N} ${BD} | tail -n1)
    echo "${BD} ${T}" >> ${OUT_BD}
done
echo "Part (c) complete → ${OUT_BD}"
