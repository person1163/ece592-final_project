import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load samples (64-bit packed timestamps)
t0 = np.fromfile("timings_0.bin", dtype=np.uint64)
t1 = np.fromfile("timings_1.bin", dtype=np.uint64)

# Extract latency = end - start (lower/upper 32 bits)
start0 = t0 & 0xffffffff
end0   = t0 >> 32
lat0   = end0 - start0

start1 = t1 & 0xffffffff
end1   = t1 >> 32
lat1   = end1 - start1

# KDE using scipy (no seaborn needed)
kde0 = gaussian_kde(lat0)
kde1 = gaussian_kde(lat1)

xs = np.linspace(
    min(lat0.min(), lat1.min()),
    max(lat0.max(), lat1.max()),
    500
)

plt.figure(figsize=(12,6))
plt.plot(xs, kde0(xs), label="Elimination", alpha=0.7)
plt.plot(xs, kde1(xs), label="No-Elimination", alpha=0.7)

plt.title("Timing Distributions (Move-Elimination vs No-Elimination)")
plt.xlabel("Latency (TSC ticks)")
plt.xlim(0,50)
plt.ylabel("Density")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("kde_plot.png", dpi=300)
plt.show()
