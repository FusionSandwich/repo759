import matplotlib.pyplot as plt

# Read data
n_values = []
times_1024 = []
with open('times_1024.txt', 'r') as f:
    next(f)  # Skip header
    for line in f:
        n, t = map(float, line.split())
        n_values.append(n)
        times_1024.append(t)

times_256 = []
with open('times_256.txt', 'r') as f:
    next(f)  # Skip header
    for line in f:
        times_256.append(float(line.split()[1]))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, times_1024, label='threads_per_block = 1024', marker='o')
plt.plot(n_values, times_256, label='threads_per_block = 256', marker='s')
plt.xlabel('Matrix Size n')
plt.ylabel('Time (ms)')
plt.title('Matrix Multiplication Time vs. Matrix Size')
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('task1.pdf')
plt.close()