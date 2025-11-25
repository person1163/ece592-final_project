import subprocess
import re
import csv
import matplotlib.pyplot as plt

# 5 test scalars — you can modify these
SCALARS = [
    "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    "00000000000000000000000000000000",
    "00000000000000000000000000000001",
    "123456789ABCDEF123456789ABCDEF12",
    "80000000000000000000000000000000"
]

# Regex to extract perf results
REGEX = {
    "cycles": re.compile(r"([\d,]+)\s+cycles"),
    "instructions": re.compile(r"([\d,]+)\s+instructions"),
    "uops": re.compile(r"([\d,]+)\s+uops_issued.any"),
    "idq": re.compile(r"([\d,]+)\s+idq_uops_not_delivered.core")
}

CSV_FILE = "perf_results.csv"

def run_scalar(scalar):
    print(f"\n=== Running scalar {scalar} ===")
    # run ./sync.sh <scalar> and capture output
    result = subprocess.run(["./sync.sh", scalar], capture_output=True, text=True)
    output = result.stdout + result.stderr

    # parse fields
    parsed = {"scalar": scalar}
    for key, regex in REGEX.items():
        m = regex.search(output)
        parsed[key] = int(m.group(1).replace(",", "")) if m else None

    # derived metric: IPC
    parsed["IPC"] = parsed["instructions"] / parsed["cycles"] if parsed["cycles"] else None
    return parsed

def save_csv(results):
    keys = results[0].keys()
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV → {CSV_FILE}")

def plot_results(results):
    scalars = [r["scalar"] for r in results]
    IPCs = [r["IPC"] for r in results]
    idqs = [r["idq"] for r in results]

    plt.figure(figsize=(12, 6))
    plt.plot(scalars, IPCs, marker="o", label="IPC")
    plt.plot(scalars, idqs, marker="s", label="IDQ stalls")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Scalar (secret)")
    plt.ylabel("Metric value")
    plt.title("Side-Channel: Perf Metric Differences Across ECC Scalars")
    plt.legend()
    plt.tight_layout()
    plt.savefig("leakage_plot.png")
    print("Saved plot → leakage_plot.png")

def main():
    all_results = []
    for s in SCALARS:
        res = run_scalar(s)
        all_results.append(res)
    
    save_csv(all_results)
    plot_results(all_results)

if __name__ == "__main__":
    main()
