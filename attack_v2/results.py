#!/usr/bin/env python3
import subprocess, struct, csv, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import shutil

# ================================
# CONFIG — CHANGE ONLY THESE
# ================================
RUNS = 5  #runs this amount of times. dont know if this is actually helpful as its the same each run anyways
ITS  = 500       # <---- EDIT THIS ONLY  ## number of scalar runs in ecc

SCALARS = {
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF": "all_F",
    "00000000000000000000000000000000": "all_0",
    "00000000000000000000000000000001": "low_1",
    "123456789ABCDEF123456789ABCDEF12": "pattern",
    "80000000000000000000000000000000": "msb_1",
}

RAW_OUT = "raw_timings"
SUMMARY_CSV = "timing_summary.csv"
REPORT_FIG  = "finalchart.png"

# ---- delete & recreate folder ONCE per full Python run ----
if os.path.exists(RAW_OUT):
    shutil.rmtree(RAW_OUT)
os.makedirs(RAW_OUT)

# ================================
# HELPERS
# ================================

def run_scalar_once(shex, label, its, run_idx):
    print(f"[RUN] {label} | ITS={its} | run {run_idx}")

    # run measurement
    subprocess.run(["./sync.sh", shex, str(its)], check=True)

    # load timings.bin
    with open("timings.bin", "rb") as f:
        data = f.read()

    n = len(data) // 8
    lats = np.zeros(n, dtype=np.uint64)

    for i in range(n):
        (word,) = struct.unpack_from("<Q", data, i*8)
        start = word & 0xffffffff
        end   = (word >> 32) & 0xffffffff
        lats[i] = (end - start) & 0xffffffff

    # save raw CSV
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
    summary_rows = []
    plot_data = {}

    for shex, label in SCALARS.items():
        plot_data[label] = []

        for r in range(RUNS):
            arr = run_scalar_once(shex, label, ITS, r)
            plot_data[label].append(arr)

            summary_rows.append({
                "scalar_hex": shex,
                "label": label,
                "ITS": ITS,
                "run": r,
                "mean": float(np.mean(arr)),
                "std":  float(np.std(arr)),
            })

    # ---- write CSV ----
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, ["scalar_hex","label","ITS","run","mean","std"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[✓] Saved CSV → {SUMMARY_CSV}")

    # ============================================================
    #            CLEAN 3-PLOT REPORT (BEST VISUALIZATION)
    # ============================================================
    labels = list(SCALARS.values())
    plt.figure(figsize=(14, 18))

    # ============================================================
    # (1) KDE Density Curves (BEST FOR LEAKAGE)
    # ============================================================
    plt.subplot(3,1,1)
    for lbl in labels:
        merged = np.concatenate(plot_data[lbl])
        xs = np.linspace(np.min(merged), np.max(merged), 600)
        kde = gaussian_kde(merged)
        plt.plot(xs, kde(xs), label=lbl)

    plt.title(f"KDE Probability Density Curves (ITS={ITS})")
    plt.xlabel("Latency (cycles)")
    plt.ylabel("Density")
    plt.legend()

    # ============================================================
    # (2) CDF Curves — good for subtle differences
    # ============================================================
    plt.subplot(3,1,2)
    for lbl in labels:
        merged = np.sort(np.concatenate(plot_data[lbl]))
        y = np.linspace(0,1,len(merged))
        plt.plot(merged, y, label=lbl)

    plt.title("CDF (Cumulative Distribution Function)")
    plt.xlabel("Latency (cycles)")
    plt.ylabel("Cumulative Probability")
    plt.legend()

    # ============================================================
    # (3) Boxplot — quick distribution overview
    # ============================================================
    plt.subplot(3,1,3)
    merged_all = [np.concatenate(plot_data[l]) for l in labels]
    plt.boxplot(merged_all, labels=labels, showmeans=True)

    plt.title("Latency Distribution Per Scalar")
    plt.xlabel("Scalar Label")
    plt.ylabel("Latency (cycles)")

    plt.tight_layout()
    plt.savefig(REPORT_FIG, dpi=200)
    print(f"[✓] Saved plot → {REPORT_FIG}")


if __name__ == "__main__":
    main()
