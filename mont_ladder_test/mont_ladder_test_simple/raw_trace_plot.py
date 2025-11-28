#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import array
import os

# Usage: python3 trace.py timings.bin out.png

inp = sys.argv[1]
out = sys.argv[2]

raw = open(inp, "rb").read()

vals = array.array('Q')
vals.frombytes(raw)

lats = np.array(vals, dtype=np.uint64)

os.makedirs(os.path.dirname(out), exist_ok=True)

plt.plot(lats)
plt.xlabel("sample index")
plt.ylabel("latency (cycles)")
plt.tight_layout()
plt.savefig(out, dpi=300)
