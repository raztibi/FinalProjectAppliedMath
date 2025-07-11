import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

# Nonlinearity functions
def psi_identity(x, beta=None):
    return 1

def psi_beverton(x, beta):
    return 1 / (1 + beta * x)

def psi_ricker(x, beta):
    return np.exp(-beta * x)

def psi_schaefer(x, beta):
    return 1 - beta * x

def psi_power(x, beta):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, np.power(x, -beta), float('inf'))

def psi_depensation(x, beta):
    return x / (1 + beta * x**2)

# Dictionary of psi variants
psi_functions = {
    #"Identity (ψ(x)=1)": psi_identity,
    "Beverton-Holt": psi_beverton,
    #"Ricker": psi_ricker,
    #"Schaefer": psi_schaefer,
    #"Power": psi_power,
    #"Depensation": psi_depensation
}

# Parameters
beta = 9.09e-9
num_age_classes = 5
T = 100

# Leslie matrix setup
#fertility = np.array([0,0,0,20,50,80 ,100,120 ,140 ,160,160,160,160,160,160], dtype=float)
#survival = np.array([0.2,0.4,0.6,0.75,0.8,0.85,0.85,0.8,0.7,0.7,0.7,0.7,0.7,0.7], dtype=float)
fertility = np.array([2,7,12,16,18])
survival = np.array([0.8,0.8,0.8,0.8])

# Plot setup
n_plots = len(psi_functions)
n_cols = 2
n_rows = (n_plots + n_cols - 1) // n_cols
fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axes = axes_arr.flatten()

for idx, (name, psi) in enumerate(psi_functions.items()):
    x = np.zeros((num_age_classes, T + 1))
    x[:, 0] = [1000, 0, 0, 0, 0]  # start with reasonable population
    eggs = np.zeros(T + 1)

    for t in range(1,T+1):
        new_x = np.zeros(num_age_classes)

        # Nonlinear recruitment: ψ depends on x₀(t)
        eggs[t] = fertility @ x[:, t-1]
        nonlinear_factor = psi(eggs[t], beta)
        new_x[0] = nonlinear_factor * eggs[t]

        # Linear survival
        for i in range(1, num_age_classes):
            new_x[i] = survival[i - 1] * x[i - 1, t-1]
        
        new_x = np.clip(new_x, 0, None)
        x[:, t] = new_x

    #axes[idx].plot(range(T + 1), eggs, label='Eggs (n₀)', linestyle='--',color='black')
    for i in range(num_age_classes):
        axes[idx].plot(range(T + 1), x[i], label=f'Age class {i+1}')
    
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Time")
    axes[idx].set_ylabel("Population")
    axes[idx].grid(True)
    axes[idx].legend()
    
# df = pd.DataFrame(x.T, columns=[f'Age_{i+1}' for i in range(num_age_classes)])
# df['Total'] = df.sum(axis=1)
# df['psi(n1)'] = [psi(x[0, t], beta) for t in range(T + 1)]

# # Export to Excel
# df.to_excel("beverton_holt_simulation.xlsx", index_label="Time")
# print("file saved: beverton_holt_simulation.xlsx")

# Hide unused subplots
for i in range(n_plots, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.suptitle("Leslie Model with Nonlinear ψ(x₀) on Recruitment", fontsize=16, y=1.03)
plt.show()

# Print final values
S = np.ones(num_age_classes)
for i in range(1, num_age_classes):
    S[i] = S[i - 1] * survival[i - 1]
R0 = np.sum(fertility * S)
n1_star = (R0 - 1) / beta
n_star = S * n1_star / R0

print("\n🔢 Predicted steady state values:")
print(f"R0 = {R0:.4f}")
print(f"n0* = {n1_star:.4f}")
print(f"Theoretical values: {n_star.round(2)}")
print(f"Simulated final values:{x[:, -1].round(2)}")

