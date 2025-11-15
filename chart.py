#!/usr/bin/env python3

import pandas as pd
import matplotlib
# Non-GUI backend for SSH / WSL / headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    csv_path = "move_elim_trials.csv"  # change if needed
    df = pd.read_csv(csv_path)

    # Quick sanity check in terminal
    print("Loaded data with columns:", list(df.columns))
    print(df.head())

    # Clean up the "type" column just in case
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    # Split into high / low
    high = df[df["type"] == "high"].copy()
    low = df[df["type"] == "low"].copy()

    if high.empty or low.empty:
        print("Warning: 'high' or 'low' rows are empty after filtering.")
        print(df["type"].value_counts())
        return

    # Sort by trial so plots look sane
    high.sort_values("trial", inplace=True)
    low.sort_values("trial", inplace=True)

    # Merge high/low on trial to compute ratios
    merged = high[["trial", "cycles", "int_eliminated", "int_not_eliminated"]].merge(
        low[["trial", "cycles", "int_eliminated", "int_not_eliminated"]],
        on="trial",
        suffixes=("_high", "_low"),
    )
    merged["cycle_ratio_high_low"] = merged["cycles_high"] / merged["cycles_low"]

    # ---- Single figure with subplots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # 1) Cycles over trials: high vs low
    ax1.plot(high["trial"], high["cycles"], label="high (no move elim)")
    ax1.plot(low["trial"], low["cycles"], label="low (with move elim)")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Cycles")
    ax1.set_title("Cycles per Trial")
    ax1.legend()

    # 2) High trials: eliminated vs not eliminated
    ax2.plot(high["trial"], high["int_eliminated"], label="high: int_eliminated")
    ax2.plot(high["trial"], high["int_not_eliminated"], label="high: int_not_eliminated")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Integer instructions")
    ax2.set_title("High (no move elim) – eliminated vs not")
    ax2.legend()

    # 3) Low trials: eliminated vs not eliminated
    ax3.plot(low["trial"], low["int_eliminated"], label="low: int_eliminated")
    ax3.plot(low["trial"], low["int_not_eliminated"], label="low: int_not_eliminated")
    ax3.set_xlabel("Trial")
    ax3.set_ylabel("Integer instructions")
    ax3.set_title("Low (with move elim) – eliminated vs not")
    ax3.legend()

    # 4) Cycle ratio high/low
    ax4.plot(merged["trial"], merged["cycle_ratio_high_low"])
    ax4.set_xlabel("Trial")
    ax4.set_ylabel("High / Low cycles")
    ax4.set_title("Cycle ratio: no-elim / elim")

    fig.suptitle("Move Elimination vs Non-Eliminated Chain (Summary)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_name = "move_elim_summary.png"
    fig.savefig(out_name)
    print(f"Saved: {out_name}")


if __name__ == "__main__":
    main()
