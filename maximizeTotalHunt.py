import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
import matplotlib.pyplot as plt


beta = 0.01
N = 5
# Leslie matrix setup
f = np.array([0,3,3,3,4], dtype=float)
s = np.array([0.6,0.7,0.7,0.7], dtype=float)

results = []

for k in range(0, N): 

    A = np.prod(s[:k]) if k > 0 else 1.0

    def R0_Hunt(h):
        h_array = np.ones(N)
        h_array[k:] = (1 - h) ** np.arange(0, N - k)

        prod_s = np.ones(N)
        for i in range(1, N):
            prod_s[i] = prod_s[i - 1] * s[i - 1]
        return np.sum(f * h_array * prod_s)

    def S(h):
        total = 0
        for i in range(k, N):
            exponent = i - k
            prod_s = np.prod(s[k:i]) if i > k else 1
            total += (1 - h)**exponent * prod_s
        return total

    def F(h):
        if h <= 0 or h > 1:
            return 0
        R0 = R0_Hunt(h)
        if R0 < 0:
            return 0
        return (A*h/ beta) * ((R0 - 1) / R0) * S(h)

    res = minimize_scalar(lambda h: -F(h), bounds=(0, 1), method='bounded')
    h_max = res.x
    F_max = -res.fun
    R0_value = R0_Hunt(h_max)
    results.append((k+1, h_max, F_max, R0_value)) #we want k to match k in the theortical values

# Create a DataFrame with results
df = pd.DataFrame(results, columns=["k", "Optimal h", "Maximum F(h)","R0_hunt"])
print(df)

# Plot Maximum Yield vs k
plt.figure(figsize=(10, 5))
plt.plot(df["k"], df["Maximum F(h)"], marker='o')
plt.title("Maximum Yield F(h) vs. Age of First Hunt (k)")
plt.xlabel("Age of First Hunt (k)")
plt.ylabel("Maximum F(h)")
plt.grid(True)
plt.show()

# Plot Optimal h vs k
plt.figure(figsize=(10, 5))
plt.plot(df["k"], df["Optimal h"], marker='s', color='orange')
plt.title("Optimal Harvest Rate h vs. Age of First Hunt (k)")
plt.xlabel("Age of First Hunt (k)")
plt.ylabel("Optimal h")
plt.grid(True)
plt.show()