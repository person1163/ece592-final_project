#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import array

# Usage: python3 hist0.py timings_0.bin out.png

inp = sys.argv[1]
out = sys.argv[2]

raw = open(inp, "rb").read()

vals = array.array('Q')
vals.frombytes(raw)

lats = np.array(vals, dtype=np.uint64)

plt.hist(lats, bins=20000)
plt.xlim(0, .03e7)
plt.ylim(4.724e8, 4.725e8)
plt.xlabel("latency (cycles)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(out, dpi=300)
