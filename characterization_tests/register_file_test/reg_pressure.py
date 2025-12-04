import pandas as pd
import matplotlib.pyplot as plt

# load CSV
df = pd.read_csv("/Users/varun/Documents/NC_State_work/Class/ECE592-Microarch Sec/Homework/final_proj/characterization_tests/register_file_test/reg_pressure_results_1r.csv")

# extract columns
x = df["P"]
y = df["time_mean"]

# plot
plt.figure(figsize=(8,5))

plt.plot(x, y, marker='o')

plt.xlabel("Live Registers")
plt.ylabel("Latency")
plt.grid(True)
plt.tight_layout()
plt.show()
