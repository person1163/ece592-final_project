import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

plt.style.use("seaborn-v0_8")   # nicer look

# Load data
t0 = np.fromfile("timings_0.bin", dtype=np.uint64)
t1 = np.fromfile("timings_1.bin", dtype=np.uint64)

lat0 = (t0 >> 32) - (t0 & 0xffffffff)
lat1 = (t1 >> 32) - (t1 & 0xffffffff)

# Clip extreme outliers
limit = max(np.percentile(lat0, 99), np.percentile(lat1, 99))
lat0 = lat0[lat0 <= limit]
lat1 = lat1[lat1 <= limit]

xmin = min(lat0.min(), lat1.min())
xmax = limit

# Moving average function
def moving_avg(x, w=200):
    return np.convolve(x, np.ones(w)/w, mode="valid")


# -----------------------------------------------------
# Build combined figure
# -----------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(20, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# -----------------------------------------------------
# 1. HISTOGRAM
# -----------------------------------------------------
ax[0,0].hist(lat0, bins=200, alpha=0.55, label="Victim=0 (Eliminated MOVs – FAST)", color="#4C72B0")
ax[0,0].hist(lat1, bins=200, alpha=0.55, label="Victim=1 (Not Eliminated – SLOW)", color="#DD8452")

ax[0,0].set_title("Distribution of Attacker Timing Samples (RDTSC deltas)", fontsize=16)
ax[0,0].set_xlabel("Measured Latency (RDTSC cycles)", fontsize=14)
ax[0,0].set_ylabel("Sample Count", fontsize=14)
ax[0,0].legend()
ax[0,0].set_xlim(xmin, xmax)

# -----------------------------------------------------
# 2. KDE Density Curve
# -----------------------------------------------------
xs = np.linspace(xmin, xmax, 700)
kde0 = gaussian_kde(lat0)
kde1 = gaussian_kde(lat1)

ax[0,1].plot(xs, kde0(xs), linewidth=2.5, label="Victim=0 (FAST)", color="#4C72B0")
ax[0,1].plot(xs, kde1(xs), linewidth=2.5, label="Victim=1 (SLOW)", color="#DD8452")

ax[0,1].set_title("Statistical Timing Profile (KDE of RDTSC deltas)", fontsize=16)
ax[0,1].set_xlabel("Measured Latency (RDTSC cycles)", fontsize=14)
ax[0,1].set_ylabel("Relative Probability Density", fontsize=14)
ax[0,1].legend()
ax[0,1].set_xlim(xmin, xmax)

# -----------------------------------------------------
# 3. Scatter Plot
# -----------------------------------------------------
ax[1,0].scatter(range(len(lat0)), lat0, s=2, alpha=0.6, label="Victim=0 (FAST)", color="#4C72B0")
ax[1,0].scatter(range(len(lat1)), lat1, s=2, alpha=0.6, label="Victim=1 (SLOW)", color="#DD8452")

ax[1,0].set_title("Raw Attacker Timing Samples Over Time", fontsize=16)
ax[1,0].set_xlabel("Sample Index (Attacker Iteration #)", fontsize=14)
ax[1,0].set_ylabel("Measured Latency (RDTSC cycles)", fontsize=14)
ax[1,0].legend()

# -----------------------------------------------------
# 4. Moving Average (200 sample window)
# -----------------------------------------------------
ax[1,1].plot(moving_avg(lat0), linewidth=2.0, label="Victim=0 (FAST)", color="#4C72B0")
ax[1,1].plot(moving_avg(lat1), linewidth=2.0, label="Victim=1 (SLOW)", color="#DD8452")

ax[1,1].set_title("Low-Pass Filtered Timing Trend (200-sample window)", fontsize=16)
ax[1,1].set_xlabel("Sample Index (Attacker Iteration #)", fontsize=14)
ax[1,1].set_ylabel("Smoothed Latency (RDTSC cycles)", fontsize=14)
ax[1,1].legend()

plt.savefig("plots.png", dpi=320)
plt.show()
