#!/usr/bin/env python3
import matplotlib.pyplot as plt
import csv

# files produced by your Slurm run
files = {
    1024: "times_1024.csv",
    512: "times_512.csv",
}

plt.figure()
for threads, fname in files.items():
    Ns, times = [], []
    with open(fname, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Ns.append(int(row["N"]))
            times.append(float(row["time_ms"]))
    plt.loglog(Ns, times, marker="o", label=f"threads/block = {threads}")

plt.xlabel("Array length $N$")
plt.ylabel("Time (ms)")
plt.title("Reduction time vs $N$")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("task2.pdf")
print("Saved plot to task2.pdf")
plt.show()
