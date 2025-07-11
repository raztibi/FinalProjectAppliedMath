import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
beta = 9.09e-9 
num_age_classes = 5
T = 100
fertility = np.array([2,7,12,16,18])
survival = np.array([0.8,0.8,0.8,0.8])
h = np.array([0, 0.2, 0.3, 0.4, 0.5])

# === Beverton-Holt ===
def psi_beverton(x, beta):
    return 1 / (1 + beta * x)

# === S' calculation ===
S_prime = np.ones(num_age_classes)
for i in range(1,num_age_classes):
    S_prime[i] = S_prime[i - 1] * survival[i-1] * (1 - h[i-1])

print("S_prime: " + str(S_prime))

# === yield theoretical function ===
def compute_equilibrium_yield():
    R0_hunt = np.sum(fertility * S_prime)
    if R0_hunt <= 1:
        return 0
    n0_star = (R0_hunt - 1) / beta
    Y = (n0_star / R0_hunt) * np.sum(h * S_prime)
    n_star = (S_prime / R0_hunt) * n0_star
    print("\n📊 Steady State (Theoretical) Values:")
    print(f"R0 with hunting: {R0_hunt:.4f}")
    print(f"n0* (eggs at equilibrium): {n0_star:.2f}")
    for i in range(num_age_classes):
        print(f"n{i+1}* = {n_star[i]:.2f}")
    return Y

# === Simulation ===
x = np.zeros((num_age_classes, T + 1))
x[:, 0] = [1000, 0, 0, 0, 0]
eggs = np.zeros(T + 1)
hunt_per_step = np.zeros(T + 1)

for t in range(1, T + 1):
    new_x = np.zeros(num_age_classes)
    eggs[t] = fertility @ x[:, t - 1]
    psi = psi_beverton(eggs[t], beta)
    new_x[0] = psi * eggs[t]

    for i in range(1, num_age_classes):
        new_x[i] = survival[i-1] * (1-h[i-1]) * x[i - 1, t - 1]
    
    new_x = np.clip(new_x, 0, None)
    x[:, t] = new_x
    hunt_per_step[t] = np.sum(h*x[:, t-1])

# === Compare therotical to simulated
Y_star = compute_equilibrium_yield()
Y_simulated = hunt_per_step[-1]

# === plot
fig, axs_arr = plt.subplots(2,2, figsize=(18, 10))
axs = axs_arr.flatten()

# eggs + population
#axs[0].plot(range(T + 1), eggs, label='Eggs (n₀)', linestyle='--', color='black')
for i in range(num_age_classes):
    axs[0].plot(range(T + 1), x[i], label=f'Age class {i+1}')
axs[0].set_ylabel("Population Size")
axs[0].set_title("Population and Egg Dynamics")
axs[0].legend()
axs[0].grid(True)

# yield and equilibrium plot
max_value = np.max(hunt_per_step)
max_time = np.argmax(hunt_per_step)
axs[1].plot(max_time, max_value, 'ro')
axs[1].annotate(f"Max: {max_value}", xy=(max_time+5, max_value))
axs[1].plot(range(T + 1), hunt_per_step, color='blue', label='Total Hunt')
axs[1].axhline(Y_star, color='red', linestyle='--', label=f'Equilibrium Yield ≈ {Y_star:.2f}')
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Total Hunt Per Step")
axs[1].set_title("Total Hunt per Time Step")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
