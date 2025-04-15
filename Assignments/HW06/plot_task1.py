import matplotlib.pyplot as plt
import numpy as np
import sys
import os # Import os to check file existence

# Define the threads per block values used in run_task1.sh
TPB1 = 1024
TPB2 = 256 # Make sure this matches the value in run_task1.sh

# Input data files (expected in the same directory as this script)
data_file1 = f"task1_times_{TPB1}.dat"
data_file2 = f"task1_times_{TPB2}.dat"
plot_file = "task1.pdf"

# Check if data files exist before attempting to load
if not os.path.exists(data_file1):
    print(f"Error: Data file {data_file1} not found.")
    print("Did you transfer it from Euler after running the Slurm job?")
    sys.exit(1)
if not os.path.exists(data_file2):
    print(f"Error: Data file {data_file2} not found.")
    print("Did you transfer it from Euler after running the Slurm job?")
    sys.exit(1)

try:
    # Read data, skipping header row (comments='#')
    data1 = np.loadtxt(data_file1, comments='#')
    data2 = np.loadtxt(data_file2, comments='#')
except Exception as e:
    print(f"An error occurred reading data: {e}")
    sys.exit(1)

# Check if data arrays are empty or have incorrect shape AFTER loading
if data1.size == 0 or (data1.ndim == 2 and data1.shape[1] != 2) or (data1.ndim == 1 and data1.shape[0] < 2):
     print(f"Error: Data file {data_file1} is empty or has incorrect format after loading.")
     sys.exit(1)
if data2.size == 0 or (data2.ndim == 2 and data2.shape[1] != 2) or (data2.ndim == 1 and data2.shape[0] < 2):
     print(f"Error: Data file {data_file2} is empty or has incorrect format after loading.")
     sys.exit(1)

# Handle cases where loadtxt might return a 1D array if only one data row exists
if data1.ndim == 1:
    data1 = data1.reshape(1, -1) # Reshape to 2D if it's 1D
if data2.ndim == 1:
    data2 = data2.reshape(1, -1) # Reshape to 2D if it's 1D


# Extract N and Time
n_values1 = data1[:, 0]
time_values1 = data1[:, 1]

n_values2 = data2[:, 0]
time_values2 = data2[:, 1]

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(n_values1, time_values1, marker='o', linestyle='-', label=f'Threads/Block = {TPB1}')
plt.plot(n_values2, time_values2, marker='s', linestyle='--', label=f'Threads/Block = {TPB2}')

# Use a logarithmic scale for the x-axis as N grows exponentially
plt.xscale('log', base=2)
# Optional: Use a logarithmic scale for the y-axis if time varies widely
# plt.yscale('log')

plt.xlabel('Matrix Size N (log scale, base 2)')
plt.ylabel('Time (milliseconds)')
plt.title('CUDA Matrix Multiplication Time vs. Matrix Size')
# Ensure x-ticks match the powers of 2 used
xticks = [2**p for p in range(5, 15)]
plt.xticks(xticks, [str(x) for x in xticks], rotation=45)

plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout() # Adjust layout to prevent labels overlapping

# Save the plot
try:
    plt.savefig(plot_file)
    print(f"Plot saved as {plot_file}")
except Exception as e:
    print(f"Error saving plot: {e}")
    sys.exit(1)

# Optionally display the plot if run interactively
# plt.show()