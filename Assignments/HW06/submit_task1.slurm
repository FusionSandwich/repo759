#!/bin/bash
#SBATCH --job-name=run_plot_task1  # Adjusted job name
#SBATCH --output=run_plot_task1_%j.out # Adjusted output file
#SBATCH --error=run_plot_task1_%j.err  # Adjusted error file
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=instruction

echo "Starting Slurm job: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Job submitted from directory: $SLURM_SUBMIT_DIR"

# Navigate to the submission directory
cd $SLURM_SUBMIT_DIR

# --- Run the data collection script ---
echo "Running data collection script run_task1.sh..."
bash ./run_task1.sh
# Capture exit status
run_status=$?
if [ $run_status -ne 0 ]; then
    echo "Error: run_task1.sh exited with status $run_status"
    # Decide if you want to exit the slurm script or attempt plotting anyway
    # exit $run_status # Uncomment to exit if run fails
else
    echo "run_task1.sh completed successfully."
fi

# --- Generate the plot ---
echo "Attempting to generate plot..."
# Load Python module (adjust if necessary for Euler's environment)
module load python
if [ $? -ne 0 ]; then
    echo "Error: Failed to load python module."
    # Plotting will likely fail, but continue to show run results if any
else
     echo "Running plotting script plot_task1.py..."
     python plot_task1.py
     plot_status=$?
     if [ $plot_status -ne 0 ]; then
         echo "Error: plot_task1.py exited with status $plot_status"
     else
         echo "plot_task1.py completed successfully. Plot saved as task1.pdf"
     fi
fi


echo "Slurm job finished."
# Use the overall status of the run script as the final exit code
exit $run_status