#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = np.loadtxt("times.txt", skiprows=1)
n_values = data[:, 0]
times = data[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(n_values, times, marker="o", linestyle="-")
plt.xlabel("Array Size (n) log(2) scale")
plt.yscale("log", base=10)
plt.ylabel("Time (milliseconds) log scale")
plt.title("Scaling Analysis of Inclusive Scan")
plt.xscale("log", base=2)
plt.grid(True)
plt.savefig("task1.pdf")
plt.show()
