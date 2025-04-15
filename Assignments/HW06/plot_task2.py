import matplotlib.pyplot as plt
import numpy as np
import os
import re # Regular expressions for parsing

def parse_time_from_log(filename):
    """Parses the execution time from a log file.
       Looks for the line 'N_TIME <n> <time_ms>'
    """
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Use regex to find the specific line format for easier parsing
                match = re.match(r"N_TIME\s+(\d+)\s+([\d.]+)", line)
                if match:
                    # n_val = int(match.group(1)) # Optional: verify n
                    time_ms = float(match.group(2))
                    return time_ms
                # Fallback for the 'Time: ... ms' line if needed
                # if line.strip().startswith("Time:"):
                #     parts = line.split()
                #     if len(parts) >= 2:
                #         return float(parts[1])
    except FileNotFoundError:
        print(f"Warning: Log file not found: {filename}")
        return None
    except Exception as e:
        print(f"Warning: Could not parse file {filename}: {e}")
        return None
    # If the specific line is not found
    print(f"Warning: Time information not found in the expected format in {filename}")
    return None

def main():
    # --- Configuration ---
    threads_per_block_values = [1024, 512] # Match the values used in Slurm script
    n_powers = range(10, 30) # 2^10 to 2^29
    n_values = [2**i for i in n_powers]

    plot_data = {} # Dictionary to store results: {tpb: {'n': [], 'time': []}}

    # --- Data Extraction ---
    for tpb in threads_per_block_values:
        print(f"Processing results for TPB = {tpb}")
        output_dir = f"output_tpb{tpb}"
        times = []
        valid_n = []
        for n in n_values:
            log_file = os.path.join(output_dir, f"n_{n}.log")
            time_ms = parse_time_from_log(log_file)
            if time_ms is not None:
                times.append(time_ms)
                valid_n.append(n)
            else:
                print(f"  Skipping n={n} for TPB={tpb} due to missing/invalid data.")

        if valid_n: # Only add if we found some data
             plot_data[tpb] = {'n': valid_n, 'time': times}
        else:
             print(f"Warning: No valid data found for TPB = {tpb}")


    # --- Plotting ---
    if not plot_data:
        print("Error: No data to plot. Exiting.")
        return

    plt.figure(figsize=(10, 6))

    for tpb, data in plot_data.items():
        if data['n']: # Check again if list is not empty
            plt.plot(data['n'], data['time'], marker='o', linestyle='-', label=f'TPB = {tpb}')

    plt.xscale('log', base=2) # Log scale for n (powers of 2)
    plt.yscale('log')         # Log scale for time often makes trends clearer
    plt.xlabel('Input Size n (log scale base 2)')
    plt.ylabel('Execution Time (milliseconds, log scale)')
    plt.title('1D Stencil Computation Time vs. Input Size')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6) # Add grid for better readability

    # Ensure x-ticks show powers of 2 nicely if possible
    plt.xticks(n_values, [f'$2^{{{p}}}$' for p in n_powers], rotation=45, ha='right')
    plt.minorticks_off() # Turn off minor ticks on x-axis if labels overlap too much
    plt.tight_layout() # Adjust layout


    # Save the plot
    plot_filename = 'task2.pdf'
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    # Optionally display the plot
    # plt.show()

if __name__ == "__main__":
    main()