import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Usage check
if len(sys.argv) != 2:
    print("Usage: python plot_results.py timing_data.csv")
    sys.exit(1)

# Get input file name
input_file = sys.argv[1]

try:
    # Read the timing data
    data = pd.read_csv(input_file)

    # Convert "NA" strings to NaN for proper plotting
    data = data.replace("NA", np.nan)

    # Convert columns to numeric
    data["N"] = pd.to_numeric(data["N"])
    data["time_1024"] = pd.to_numeric(data["time_1024"])
    data["time_256"] = pd.to_numeric(data["time_256"])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        data["N"], data["time_1024"], "o-", label="threads_per_block = 1024"
    )
    plt.plot(
        data["N"], data["time_256"], "s-", label="threads_per_block = 256"
    )
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Array Size (N)")
    plt.ylabel("Time (ms)")
    plt.title("Parallel Reduction Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("task2.pdf")
    print("Plot saved as task2.pdf")

    # Display the plot
    plt.show()

except Exception as e:
    print(f"Error processing file {input_file}: {e}")
    sys.exit(1)
