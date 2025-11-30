import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/home/vcvenkat/ece592-final_project/portsmash/secp384r1_all_1000/results_summary.csv")

# Experiment groups and their corresponding columns
groups = {
    "baseline": ("baseline_elim", "baseline_notelim"),
    "openssl": ("openssl_elim", "openssl_notelim"),
    "add": ("add_elim", "add_notelim"),
    "double": ("double_elim", "double_notelim"),
    "mul": ("mul_elim", "mul_notelim"),
    "ad": ("ad_elim", "ad_notelim"),
}

for name, (elim_col, notelim_col) in groups.items():
    plt.figure(figsize=(6,4))
    plt.hist(df[elim_col], bins=20, alpha=0.6, label=f"{name} elim")
    plt.hist(df[notelim_col], bins=20, alpha=0.6, label=f"{name} notelim")
    plt.title(f"{name} histogram")
    plt.xlabel("count")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
