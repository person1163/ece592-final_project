#!/usr/bin/env python3

import pandas as pd
import matplotlib
matplotlib.use("Agg")    # safe for headless nodes
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results.csv")

    cases = df["case"]
    metrics = [
        ("cycles", "Cycles"),
        ("instructions", "Instructions"),
        ("uops_issued", "Uops Issued"),
        ("uops_executed", "Uops Executed (core)"),
        ("idq_not_delivered", "IDQ Uops Not Delivered"),
        ("resource_stalls", "Resource Stalls (RS)"),
        ("time_sec", "Time (sec)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        ax.bar(cases, df[col], color=["#3b82f6", "#ef4444", "#22c55e"])
        ax.set_title(title)
        ax.set_xticklabels(cases, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Victim Performance Comparison: Move Elimination vs No-Elimination Attackers", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = "victim_attack_summary.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
