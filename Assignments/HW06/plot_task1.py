import matplotlib.pyplot as plt
import numpy as np
import sys

# Define the threads per block values used in run_task1.sh
TPB1 = 1024
TPB2 = 512 # Make sure this matches the value in run_task1.sh

# Input data files
data_file1 = f"task1_times_{TPB1}.dat"
data_file2 = f"task1_times_{TPB2}.dat"
plot_file = "task1.pdf"

try:
    # Read data, skipping header row
    data1 = np.loadtxt(data_file1, comments='#')
    data2 = np.loadtxt(data_file2, comments='#')
except IOError as e:
    print(f"Error reading data files: {e}")
    print("Did you run ./run_task1.sh first?")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred reading data: {e}")
    sys.exit(1)

# Check if data arrays are empty or have incorrect shape
if data1.size == 0 or data1.ndim != 2 or data1.shape[1] != 2:
     print(f"Error: Data file {data_file1} is empty or has incorrect format.")
     sys.exit(1)
if data2.size == 0 or data2.ndim != 2 or data2.shape[1] != 2:
     print(f"Error: Data file {data_file2} is empty or has incorrect format.")
     sys.exit(1)


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

plt.xlabel('Matrix Size N (log scale)')
plt.ylabel('Time (milliseconds)')
plt.title('CUDA Matrix Multiplication Time vs. Matrix Size')
plt.legend()
plt.grid(True, which="both", ls="--")

# Save the plot
try:
    plt.savefig(plot_file)
    print(f"Plot saved as {plot_file}")
except Exception as e:
    print(f"Error saving plot: {e}")
    sys.exit(1)

# Optionally display the plot if run interactively
# plt.show()