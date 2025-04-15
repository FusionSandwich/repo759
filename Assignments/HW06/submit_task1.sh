#!/bin/bash
#SBATCH --job-name=run_task1_data # Adjusted job name slightly
#SBATCH --output=run_task1_data_%j.out # Adjusted output file
#SBATCH --error=run_task1_data_%j.err  # Adjusted error file
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction

echo "Starting Slurm job: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Job submitted from directory: $SLURM_SUBMIT_DIR"
echo "Purpose: Compile CUDA code and generate timing data files."

# Navigate to the submission directory
cd $SLURM_SUBMIT_DIR

# --- Run the data collection script ---
echo "Running data collection script run_task1.sh..."
bash ./run_task1.sh
# Capture exit status
run_status=$?
if [ $run_status -ne 0 ]; then
    echo "Error: run_task1.sh exited with status $run_status. Data generation may be incomplete."
else
    echo "run_task1.sh completed successfully. Check $SLURM_SUBMIT_DIR for .dat files."
fi

# --- Plotting section removed ---
# Plotting should be done manually after transferring .dat files from Euler

echo "Slurm job finished."
# Use the overall status of the run script as the final exit code
exit $run_status