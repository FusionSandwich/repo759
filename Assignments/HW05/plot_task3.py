#!/usr/bin/env python3
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Define exponents from 10 to 29 so that n = 2^10, 2^11, …, 2^29.
exponents = range(10, 30)
n_values = [2**e for e in exponents]

# Two thread configurations: 512 and 16 threads per block.
threads_configs = [512, 16]

# Dictionary to store the execution times for each configuration.
results = {512: [], 16: []}

for threads in threads_configs:
    for n in n_values:
        try:
            # Run the task3 executable with arguments: n and threads
            # Capture its output (which we assume prints the elapsed kernel time in ms as the first line).
            output = subprocess.check_output(["./task3", str(n), str(threads)], universal_newlines=True)
            # Parse the output: first line is the kernel time in ms.
            lines = output.strip().splitlines()
            if len(lines) > 0:
                time_ms = float(lines[0])
            else:
                time_ms = np.nan
        except Exception as e:
            print(f"Error running task3 with n={n} and threads={threads}: {e}")
            time_ms = np.nan

        results[threads].append(time_ms)
        print(f"n={n}, threads={threads}, time={time_ms} ms")

# Create the plot.
plt.figure(figsize=(8,6))
plt.plot(n_values, results[512], marker='o', label="512 threads per block")
plt.plot(n_values, results[16], marker='s', label="16 threads per block")
plt.xlabel("n (array size)")
plt.ylabel("Kernel Execution Time (ms)")
plt.title("vscale Kernel Execution Time vs n")
plt.legend()
plt.xscale('log', basex=2)  # Use a logarithmic x-axis (base 2) since n are powers of 2.
plt.grid(True, which="both", linestyle="--")
plt.savefig("task3.pdf")
plt.show()
