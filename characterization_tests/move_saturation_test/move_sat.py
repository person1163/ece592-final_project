import pandas as pd
import matplotlib.pyplot as plt

# load CSV
df = pd.read_csv("/Users/varun/Documents/NC_State_work/Class/ECE592-Microarch Sec/Homework/final_proj/characterization_tests/move_saturation_test/move_saturation_raw_iters_f.csv")

# group by K and average the 10 iterations
g = df.groupby("K").mean(numeric_only=True)

# extract x and y
x = g.index
elim = g["elim"]
not_elim = g["not_elim"]

# plot
plt.figure(figsize=(8,5))

plt.plot(x, elim, marker='o', label="elim_mean")
plt.plot(x, not_elim, marker='o', label="not_elim_mean")

plt.xlabel("Number of Moves")
plt.ylabel("Moves")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
