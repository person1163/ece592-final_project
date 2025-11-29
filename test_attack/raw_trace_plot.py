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

# Convert to numpy
words = np.array(vals, dtype=np.uint64)

# Decode timestamps:
#   low  32 bits = start
#   high 32 bits = end
start =  words        & 0xffffffff
end   = (words >> 32) & 0xffffffff

# Compute actual latency (cycles)
lats = (end - start).astype(np.int64)

# Create output directory if needed
outdir = os.path.dirname(out)
if outdir:
    os.makedirs(outdir, exist_ok=True)

plt.plot(lats)
plt.xlabel("sample index")
plt.ylabel("latency (cycles)")
plt.tight_layout()
plt.savefig(out, dpi=300)
