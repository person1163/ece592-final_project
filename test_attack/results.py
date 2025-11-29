#!/usr/bin/env python3
import subprocess, struct, csv, os, shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ================================
# CONFIG — EDIT THESE
# ================================
RUNS = 5

# victim modes:
#   0 → memcpy/bulk path
#   1 → rename-heavy register path
SCALARS = {
    "0": "memcpy_mode",
    "1": "rename_heavy_mode",
}

RAW_OUT = "raw_timings"
SUMMARY_CSV = "timing_summary.csv"
REPORT_FIG  = "final_leakage_report.png"

# recreate folder
if os.path.exists(RAW_OUT):
    shutil.rmtree(RAW_OUT)
os.makedirs(RAW_OUT)


# ================================
# RUN + LOAD RAW TIMINGS
# ================================
def run_scalar_once(mode, label, run_idx):
    print(f"[RUN] mode={mode} | {label} | run {run_idx}")

    # run victim + attacker
    subprocess.run(["./sync.sh", mode], check=True)

    # read timings.bin
    with open("timings.bin", "rb") as f:
        data = f.read()

    n = len(data) // 8
    lats = np.zeros(n, dtype=np.uint64)

    for i in range(n):
        (word,) = struct.unpack_from("<Q", data, i*8)
        start = word & 0xffffffff
        end   = (word >> 32) & 0xffffffff
        lats[i] = (end - start) & 0xffffffff

    # save CSV
    out_csv = f"{RAW_OUT}/{label}_run{run_idx}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx","latency"])
        w.writerows(enumerate(lats))

    return lats


# ================================
# MAIN
# ================================
def main():
    all_data = {}

    # collect all runs
    for mode, label in SCALARS.items():
        all_data[label] = []
        for r in range(RUNS):
            arr = run_scalar_once(mode, label, r)
            all_data[label].append(arr)

    # merge + filter (optional)
    clean_data = {}
    for label in all_data:
        merged = np.concatenate(all_data[label])
        filtered = merged[(merged > 100) & (merged < 5000)]
        clean_data[label] = filtered

    labels = list(clean_data.keys())
    merged_all = [clean_data[l] for l in labels]

    # =========================
    #      FINAL PLOTS
    # =========================
    plt.figure(figsize=(16, 24))

    # RAW SCATTER
    plt.subplot(4,1,1)
    for label in labels:
        arr = clean_data[label]
        plt.scatter(range(len(arr)), arr, s=1, label=label)
    plt.title("RAW Timing Scatter")
    plt.xlabel("Sample Index")
    plt.ylabel("Latency (cycles)")
    plt.legend()

    # MEAN BAR CHART
    plt.subplot(4,1,2)
    mean_vals = [np.mean(clean_data[l]) for l in labels]
    plt.bar(labels, mean_vals, color="skyblue")
    plt.title("Mean Latency Per Mode")
    plt.ylabel("Mean cycles")

    # VIOLIN
    plt.subplot(4,1,3)
    plt.violinplot(merged_all, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.title("Violin Plot")
    plt.ylabel("Latency (cycles)")

    # KDE CURVES
    plt.subplot(4,1,4)
    for lbl in labels:
        d = clean_data[lbl]
        xs = np.linspace(np.min(d), np.max(d), 500)
        kde = gaussian_kde(d)
        plt.plot(xs, kde(xs), label=lbl)
    plt.title("KDE Density Curves")
    plt.xlabel("Latency (cycles)")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(REPORT_FIG, dpi=200)
    print(f"[✓] Saved report → {REPORT_FIG}")


if __name__ == "__main__":
    main()
