import pandas as pd
import matplotlib.pyplot as plt

# --- Load CSV ---
# Replace with your actual filename
df = pd.read_csv("/Users/varun/Documents/NC_State_work/Class/ECE592-Microarch Sec/Homework/final_proj/characterization_tests/ROB_move_elim_test/rob_pressure_results.csv")

# --- Extract columns ---
x = df["M"]
elim = df["elim_mean"]
notelim = df["notelim_mean"]

# --- Plot ---
plt.figure(figsize=(8,5))

plt.plot(x, elim, marker='o', label='elim_mean')
plt.plot(x, notelim, marker='o', label='notelim_mean')

plt.xlabel("ALU OPS in ROB")
plt.ylabel("MOVES")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
