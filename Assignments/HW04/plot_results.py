import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Data (example; replace with actual results)
schedules = ['static', 'dynamic', 'guided']
data = {}
for sched in schedules:
    threads = []
    times = []
    with open(f'results_{sched}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            threads.append(int(parts[1]))
            times.append(float(parts[3]))
    data[sched] = (threads, times)

# Plotting
with PdfPages('task4.pdf') as pdf:
    for sched in schedules:
        threads, times = data[sched]
        plt.figure(figsize=(8, 6))
        plt.plot(threads, times, marker='o', label=f'{sched} scheduling')
        plt.xlabel('Number of Threads')
        plt.ylabel('Time (ms)')
        plt.title(f'N-body Simulation Time vs. Threads ({sched.capitalize()} Scheduling)')
        plt.grid(True)
        plt.legend()
        pdf.savefig()
        plt.close()

print("Plots saved to task4.pdf")