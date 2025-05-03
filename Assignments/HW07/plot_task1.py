import numpy as np
import matplotlib.pyplot as plt

# — Part (b) plot: time vs n —
data_n = np.loadtxt("task1_times.txt", comments="#")
n = data_n[:, 0]
t = data_n[:, 1]

plt.figure()
plt.loglog(n, t, marker="o")
plt.xlabel("Matrix size $n$")
plt.ylabel("Time (ms)")
plt.title("Task1: Time vs Matrix Size\n(block_dim=16)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("time_vs_n.png", dpi=300)
print("Saved plot → time_vs_n.png")

# — Part (c) plot: time vs block_dim at n=2^14 —
data_bd = np.loadtxt("blockdim_sweep.txt", comments="#")
bd = data_bd[:, 0]
tb = data_bd[:, 1]

plt.figure()
plt.plot(bd, tb, marker="o")
plt.xlabel("Block dimension")
plt.ylabel("Time (ms)")
plt.title("Task1: Time vs Block Dim\n(n=16384)")
plt.grid(True, ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("time_vs_blockdim.png", dpi=300)
print("Saved plot → time_vs_blockdim.png")

# — Find best block_dim —
best = bd[np.argmin(tb)]
best_time = tb.min()
print(f"Best block_dim = {int(best)} (time = {best_time:.3f} ms)")
