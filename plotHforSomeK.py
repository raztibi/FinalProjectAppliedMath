import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Parameters
f = np.array([2,7,12,16,18])
s = np.array([0.8,0.8,0.8,0.8])
N = len(f)
beta = 9.09e-9

# Choose the value of k (e.g., 3)
k = 2

A = np.prod(s[:k]) if k > 0 else 1.0

# Define R0_Hunt using the corrected formula
def R0_Hunt(h):
    h_array = np.ones(N)
    h_array[k:] = (1 - h) ** np.arange(0, N - k)

    prod_s = np.ones(N)
    for i in range(1, N):
        prod_s[i] = prod_s[i - 1] * s[i - 1]
    return np.sum(f * h_array * prod_s)

# Define S(h)
def S(h):
    total = 0
    for i in range(k, N):
        exponent = i - k
        prod_s = np.prod(s[k:i]) if i > k else 1
        total += (1 - h)**exponent * prod_s
    return total

# Define F(h)
def F(h):
    if h < 0 or h > 1:
        return 0
    R0 = R0_Hunt(h)
    if R0 <= 1e-8:  # avoid division by 0 or log of zero
        return -np.nan
    value = (A*h / beta) * ((R0 - 1) / R0) * S(h)
    return value if value >= 0 else -np.nan

# Compute F(h) over a grid
h_values = np.linspace(0, 1, 300)
F_values = [F(h) for h in h_values]

# Find maximum using optimization
res = minimize_scalar(lambda h: -F(h), bounds=(1e-6, 1-1e-6), method='bounded')
h_max = res.x
F_max = -res.fun

# Plot F(h) and mark maximum point with a legend only
plt.figure(figsize=(10, 5))
plt.plot(h_values, F_values, label=f"F(h) for starting hunt at age={k+1}") #we want the k to be like in the theortical equations
plt.plot(h_max, F_max, 'ro', label=f"Max point: h={h_max:.3f}, F={F_max:.3f}")
plt.title(f"Yield F(h) vs Harvest Rate h (for age={k+1})")
plt.xlabel("Harvest Rate h")
plt.ylabel("F(h)")
plt.grid(True)
plt.legend()
plt.show()
