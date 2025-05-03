#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --output=run_task2_%j.out
#SBATCH --error=run_task2_%j.err
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=4G

module load nvidia/cuda/11.8

# Thread‐block sizes to compare
THREADS=(1024 512)

for t in "${THREADS[@]}"; do
  outfile="times_${t}.csv"
  echo "N,time_ms" > "${outfile}"
  for exp in $(seq 10 30); do
    N=$((1<<exp))
    # task2 prints: <sum>\n<ms>\n → grab only the ms
    ms=$(./task2 "${N}" "${t}" | tail -n1)
    echo "${N},${ms}" >> "${outfile}"
  done
done
