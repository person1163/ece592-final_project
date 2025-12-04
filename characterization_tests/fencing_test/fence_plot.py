import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/varun/Documents/NC_State_work/Class/ECE592-Microarch Sec/Homework/final_proj/characterization_tests/fencing_test/results.csv")

# extract groups
g00 = df[(df["opt_level"] == "O0") & (df["mode"] == 0)]["time_elapsed"]
g01 = df[(df["opt_level"] == "O0") & (df["mode"] == 1)]["time_elapsed"]
g30 = df[(df["opt_level"] == "O3") & (df["mode"] == 0)]["time_elapsed"]
g31 = df[(df["opt_level"] == "O3") & (df["mode"] == 1)]["time_elapsed"]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# O0 mode 0
axs[0,0].hist(g00, bins=1000, edgecolor='black')
axs[0,0].set_title("No Optimization(O0) Move Eliminable")
axs[0,0].set_xlabel("Time (s)")
axs[0,0].set_ylabel("Count")

# O0 mode 1
axs[0,1].hist(g01, bins=1000, edgecolor='black')
axs[0,1].set_title("No Optimization(O0) Move Not Eliminable")
axs[0,1].set_xlabel("Time (s)")
axs[0,1].set_ylabel("Count")

# O3 mode 0
axs[1,0].hist(g30, bins=1000, edgecolor='black')
axs[1,0].set_title("High Optimization(O3) Move Eliminable")
axs[1,0].set_xlabel("Time (s)")
axs[1,0].set_ylabel("Count")

# O3 mode 1
axs[1,1].hist(g31, bins=1000, edgecolor='black')
axs[1,1].set_title("High Optimization(O3) Move Not Eliminable")
axs[1,1].set_xlabel("Time (s)")
axs[1,1].set_ylabel("Count")

plt.tight_layout()
plt.show()
