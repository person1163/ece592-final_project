#!/usr/bin/env python3

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results.csv")

    # Pretty names for x-axis
    pretty_names = {
        "victim_only": "Victim Only",
        "victim_with_attacker": "Victim + High-Move Attacker\n(move eliminated)",
        "victim_with_low_moves": "Victim + Low-Move Attacker\n(no move elimination)",
    }

    df["pretty_case"] = df["case"].map(pretty_names)

    # Convert seconds -> milliseconds for better readability
    df["time_ms"] = df["time_sec"] * 1000

    # Metrics with clear Y-axis labels
    metrics = [
        ("cycles", "Cycles (CPU cycles)"),
        ("instructions", "Instructions Retired"),
        ("uops_issued", "µops Issued"),
        ("idq_not_delivered", "IDQ: µops Not Delivered to Front End"),
        ("resource_stalls", "RS Resource Stalls"),
        ("time_ms", "Execution Time (ms)"),   # <--- UPDATED
    ]

    # Make 2x3 subplot layout (6 charts)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ["#3b82f6", "#ef4444", "#22c55e"]

    for idx, (col, y_label) in enumerate(metrics):
        ax = axes[idx]
        ax.bar(df["pretty_case"], df[col], color=colors)
        ax.set_title(y_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xticklabels(df["pretty_case"], rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Victim Performance Under Move-Eliminated vs Non-Eliminated Attackers",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = "victim_attack_summary_clean.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
