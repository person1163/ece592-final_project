import numpy as np
import matplotlib.pyplot as plt

# Load samples (64-bit packed timestamps)
t0 = np.fromfile("timings_0.bin", dtype=np.uint64)
t1 = np.fromfile("timings_1.bin", dtype=np.uint64)

# Extract latency = end - start
lat0 = (t0 >> 32) - (t0 & 0xffffffff)
lat1 = (t1 >> 32) - (t1 & 0xffffffff)

plt.figure(figsize=(14,7))

bins = 2000  # increase/decrease if needed

plt.hist(lat0,
         bins=bins,
         alpha=0.6,
         label="Elimination (fast path)",
         color="tab:blue")

plt.hist(lat1,
         bins=bins,
         alpha=0.6,
         label="No-Elimination (slow path)",
         color="tab:orange")

plt.title("Timing Histogram\nMove-Elimination vs No-Elimination", fontsize=16)
plt.xlabel("Latency per Probe (TSC ticks)", fontsize=14)
plt.ylabel("Count (number of samples)", fontsize=14)

plt.grid(alpha=0.25)
plt.legend(fontsize=13)
plt.xlim(0,60)   # modify if needed
plt.tight_layout()
plt.savefig("histogram_plot.png", dpi=300)
plt.show()
