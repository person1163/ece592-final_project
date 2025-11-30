#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import array
import numpy as np
import matplotlib.pyplot as plt

CEIL = 511

def normalize(x):
    if x > CEIL: return CEIL
    if x < 0: return CEIL
    return x

try:
    fp = open(sys.argv[1], "rb")
except:
    print("Usage: python %s <timings.bin>" % (sys.argv[0]))
    sys.exit(1)

out = fp.read()
fp.close()

timings = array.array('I')
timings.frombytes(out)

lats = []
for i in range(0, len(timings), 2):
    lats.append(timings[i+1] - timings[i])

lats = list(map(normalize, lats))

plt.figure(figsize=(12,4))
plt.plot(
    lats,
    marker='o',
    ms=4.0,
    fillstyle='full',
    markeredgewidth=0.0,
    linestyle='-'
)


plt.xlabel("Sample Index")
plt.ylabel("Latency (cycles)")
plt.title("Raw Latency Trace")

plt.tight_layout()
plt.savefig("/home/vcvenkat/ece592-final_project/results_3841r1_0001/openssl.png", dpi=300)
plt.show()
