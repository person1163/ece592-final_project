#!/usr/bin/env python3
import numpy as np
import array
import sys
from scipy.stats import ttest_ind

def load(fname):
    raw = open(fname, "rb").read()
    vals = array.array('Q')
    vals.frombytes(raw)
    return np.array(vals, dtype=np.uint64)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    s = np.sqrt(((na - 1)*sa + (nb - 1)*sb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / s

f0 = sys.argv[1]
f1 = sys.argv[2]

l0 = load(f0)
l1 = load(f1)

# Welch t-test (fast, O(N))
t, p_t = ttest_ind(l0, l1, equal_var=False)

# Cohen's d (fast, O(N))
d = cohens_d(l0, l1)

# Threshold classifier (fast, O(N))
thr = (np.median(l0) + np.median(l1)) / 2
acc0 = np.mean(l0 < thr)
acc1 = np.mean(l1 > thr)
acc = 0.5 * (acc0 + acc1)

# KS test on downsampled data only (critical for speed)
step = max(1, len(l0) // 100_000)  # cap at ~100k points
l0_ds = l0[::step]
l1_ds = l1[::step]

from scipy.stats import ks_2samp
D, p_ks = ks_2samp(l0_ds, l1_ds)

print("Welch t-test: t =", t, " p =", p_t)
print("KS test (downsampled): D =", D, " p =", p_ks)
print("Cohen d:", d)
print("Classifier accuracy:", acc)
