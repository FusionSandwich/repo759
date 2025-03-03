#!/usr/bin/env python3
"""
plot_task2.py
Usage: python3 plot_task2.py <results_file> <output_png>
"""
import sys
import re
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python3 plot_task2.py <results_file> <output_png>")
    sys.exit(1)

results_file = sys.argv[1]
output_png = sys.argv[2]

threads = []
times = []

with open(results_file, 'r') as f:
    # Read and strip all non-empty lines.
    lines = [line.strip() for line in f if line.strip()]

# We expect each run to be represented by 4 consecutive lines:
#   1. Header line (e.g., "Running task2 with n=1024 and t=4")
#   2. First element (float)
#   3. Last element (float)
#   4. Runtime in ms (float)
if len(lines) % 4 != 0:
    print("Warning: The results file does not appear to be grouped in 4-line entries. Processing available lines...")

for i in range(0, len(lines), 4):
    header = lines[i]
    # Use regex to extract the thread count from the header. It looks for a pattern like "t=4"
    match = re.search(r"t\s*=\s*(\d+)", header)
    if not match:
        print(f"Warning: Could not extract thread count from header: '{header}'")
        continue
    try:
        t_val = int(match.group(1))
        # The runtime is expected on the 4th line of the group.
        runtime_line = lines[i + 3]
        runtime = float(runtime_line)
        threads.append(t_val)
        times.append(runtime)
    except (IndexError, ValueError) as e:
        print(f"Error processing group starting at line {i+1}: {e}")
        continue

if not threads or not times:
    print("No valid data found in the results file.")
    sys.exit(1)

plt.figure(figsize=(6, 4))
plt.plot(threads, times, marker='o', linestyle='-')
plt.xlabel("Number of Threads")
plt.ylabel("Convolution Time (ms)")
plt.title("Convolution Execution Time vs. Number of Threads")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_png)
print(f"Plot saved as {output_png}")
