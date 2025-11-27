#!/usr/bin/env python3
import subprocess, struct, csv, os, shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ================================
# CONFIG — CHANGE ONLY THESE
# ================================
RUNS = 5
ITS  = 20000      # <---- EDIT THIS ONLY

SCALARS = {
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF": "all_F",
    "00000000000000000000000000000000": "all_0",
    "00000000000000000000000000000001": "low_1",
    "123456789ABCDEF123456789ABCDEF12": "pattern",
    "80000000000000000000000000000000": "msb_1",
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
def run_scalar_once(shex, label, its, run_idx):
    print(f"[RUN] {label} | ITS={its} | run {run_idx}")

    # run victim + attacker
    subprocess.run(["./sync.sh", shex, str(its)], check=True)

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
    out_csv = f"{RAW_OUT}/{label}_ITS{its}_run{run_idx}.csv"
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
    for shex, label in SCALARS.items():
        all_data[label] = []
        for r in range(RUNS):
            arr = run_scalar_once(shex, label, ITS, r)
            all_data[label].append(arr)

    # merge + filter outliers
    clean_data = {}
    for label in all_data:
        merged = np.concatenate(all_data[label])
        filtered = merged[(merged > 200) & (merged < 2000)]
        clean_data[label] = filtered

    labels = list(clean_data.keys())
    merged_all = [clean_data[lbl] for lbl in labels]

    # =========================
    #      FINAL PLOTS
    # =========================
    plt.figure(figsize=(16, 24))

    # ============================================================
    # (A) RAW SCATTER (debug view)
    # ============================================================
    plt.subplot(4,1,1)
    for label in labels:
        arr = clean_data[label]
        plt.scatter(range(len(arr)), arr, s=1, label=label)

    plt.title(f"RAW Timing Scatter (filtered)  ITS={ITS}")
    plt.xlabel("Sample Index")
    plt.ylabel("Latency (cycles)")
    plt.ylim(600, 800)
    plt.legend()

    # ============================================================
    # (B) MEAN LATENCY BAR CHART (macro leakage indicator)
    # ============================================================
    plt.subplot(4,1,2)
    mean_vals = [np.mean(clean_data[lbl]) for lbl in labels]
    plt.bar(labels, mean_vals, color="skyblue")
    plt.title("Mean Latency Per Scalar (Macro Leakage Indicator)")
    plt.ylabel("Mean cycles")

    # ============================================================
    # (C) VIOLIN PLOT (BEST leakage visualization)
    # ============================================================
    plt.subplot(4,1,3)
    plt.violinplot(merged_all, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.title(f"Violin Plot — Full Distribution Per Scalar (ITS={ITS})")
    plt.ylabel("Latency (cycles)")
    plt.ylim(600, 800)

    # ============================================================
    # (D) KDE DENSITY CURVES (distribution shape)
    # ============================================================
    plt.subplot(4,1,4)
    for lbl in labels:
        data = clean_data[lbl]
        xs = np.linspace(np.min(data), np.max(data), 500)
        kde = gaussian_kde(data)
        plt.plot(xs, kde(xs), label=lbl)

    plt.title("KDE Probability Density Curves")
    plt.xlabel("Latency (cycles)")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(REPORT_FIG, dpi=200)
    print(f"[✓] Saved report → {REPORT_FIG}")


if __name__ == "__main__":
    main()
