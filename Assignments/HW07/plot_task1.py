import numpy as np
import matplotlib.pyplot as plt

# — Part (b) plot: time vs n for each data type —
data_n = np.loadtxt("task1_times.txt", comments="#")
n = data_n[:, 0]
t_times = data_n[:, 1]  # Just one time column

plt.figure(figsize=(10, 6))
plt.loglog(n, t_times, marker="o", label="Matrix Multiplication")
plt.xlabel("Matrix size $n$")
plt.ylabel("Time (ms)")
plt.title("Task1: Time vs Matrix Size\n(block_dim=16)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("time_vs_n.png", dpi=300)
print("Saved plot → time_vs_n.png")

# Calculate theoretical complexity O(n^3)
n_theory = np.logspace(np.log10(n.min()), np.log10(n.max()), 100)
# Scale to match the data
scale_factor = t_times[5] / (n[5] ** 3)  # Use middle point for scaling
t_theory = scale_factor * n_theory**3

plt.figure(figsize=(10, 6))
plt.loglog(n, t_times, marker="o", label="Measured time")
plt.loglog(n_theory, t_theory, "k--", label="$O(n^3)$ theoretical")
plt.xlabel("Matrix size $n$")
plt.ylabel("Time (ms)")
plt.title("Task1: Time vs Matrix Size with Theoretical Complexity")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("time_vs_n_theory.png", dpi=300)
print("Saved plot → time_vs_n_theory.png")

# — Part (c) plot: time vs block_dim at n=2^14 —
data_bd = np.loadtxt("blockdim_sweep.txt", comments="#")
bd = data_bd[:, 0]
tb_times = data_bd[:, 1]  # Just one time column

plt.figure(figsize=(10, 6))
plt.plot(bd, tb_times, marker="o", label="Matrix Multiplication")
plt.xlabel("Block dimension")
plt.ylabel("Time (ms)")
plt.title("Task1: Time vs Block Dim\n(n=16384)")
plt.grid(True, ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("time_vs_blockdim.png", dpi=300)
print("Saved plot → time_vs_blockdim.png")

# — Find best block_dim —
best_block = bd[np.argmin(tb_times)]
min_time = tb_times.min()

print(f"Best block_dim = {int(best_block)} (time = {min_time:.3f} ms)")

# Analysis for questions
print("\nAnswers for questions:")
print(
    "b) See generated plots for the time taken by the algorithm as a function of n."
)
print(
    f"c) The best performing value of block_dim when n=2^14 is {int(best_block)}."
)
print("d) Analysis of performance differences between data types:")
print(
    "   - Double precision (64-bit) operations are typically slower than single precision (32-bit) operations"
)
print(
    "   - This is because double precision uses twice as much memory bandwidth and has fewer operations per clock cycle"
)
print(
    "   - GPU hardware is often optimized for single precision (float) calculations"
)
print(
    "   - Specific architectures may have different ratios between int, float, and double performance"
)
print(
    "\ne) For comparison with HW06, you should compare the best runtime for n=2^14 from this task"
)
print(
    f"   with the result from HW06 matrix multiplication task. Current best time: {min_time:.3f} ms"
)
print(
    "\nf) For comparison with HW02, you should compare the best runtime for n=2^14 from this task"
)
print(
    "   with the result from HW02 (serial implementation mmul1). Likely the GPU implementation is"
)
print(
    "   significantly faster due to massive parallelism available on the GPU for matrix operations."
)
