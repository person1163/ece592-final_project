import csv
import subprocess
import re
import sys
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
BIN = Path("/mnt/ncsudrive/v/vcvenkat/temp/ece592-final_project/characterization_tests/move_saturation_test/move_fence")   # CHANGE THIS
SWEEPS = 10      # how many times to sweep K=1..MAXK
REPEATS = 1     # perf runs per (iter, K)
MAXK = 256
CSV_OUT = "/mnt/ncsudrive/v/vcvenkat/temp/ece592-final_project/characterization_tests/move_saturation_test/move_saturation_raw_iters_f.csv"

if not BIN.is_file():
    print("ERROR: binary not found:", BIN)
    sys.exit(1)

# -------------------------------------------------------
# PERF PARSING REGEX
# -------------------------------------------------------
RE_CYC   = re.compile(r"^\s*([\d,]+)\s+cycles")
RE_INST  = re.compile(r"^\s*([\d,]+)\s+instructions")
RE_ELIM  = re.compile(r"^\s*([\d,]+)\s+move_elimination\.int_eliminated")
RE_NOEL  = re.compile(r"^\s*([\d,]+)\s+move_elimination\.int_not_eliminated")
RE_TIME  = re.compile(r"([\d\.]+)\s+seconds time elapsed")

def parse_perf(text):
    cycles = inst = elim = noelim = 0
    time_elapsed = 0.0

    for line in text.splitlines():
        m = RE_CYC.search(line)
        if m:
            cycles = int(m.group(1).replace(",", ""))

        m = RE_INST.search(line)
        if m:
            inst = int(m.group(1).replace(",", ""))

        m = RE_ELIM.search(line)
        if m:
            elim = int(m.group(1).replace(",", ""))

        m = RE_NOEL.search(line)
        if m:
            noelim = int(m.group(1).replace(",", ""))

        m = RE_TIME.search(line)
        if m:
            time_elapsed = float(m.group(1))

    # print(elim)
    return cycles, inst, elim, noelim, time_elapsed

# -------------------------------------------------------
# RUN SWEEPS
# -------------------------------------------------------
print("Writing:", CSV_OUT)

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "iter",
        "K",
        "repeat",
        "cycles",
        "instructions",
        "elim",
        "not_elim",
        "time"
    ])

    for it in range(1, SWEEPS + 1):
        print(f"ITER={it}")
        for K in range(1, MAXK + 1):
            print(f"  K={K}")
            for r in range(1, REPEATS + 1):
                cmd = [
                    "taskset", "-c", "1",
                    "perf", "stat",
                    "-e", "cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated",
                    str(BIN), str(K)
                ]

                proc = subprocess.run(
                    cmd,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True
                )

                c, i, e, n, t = parse_perf(proc.stderr)
                writer.writerow([it, K, r, c, i, e, n, t])
