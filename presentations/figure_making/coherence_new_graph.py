import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns

# define center bird
bird_position = np.array([0, 0])
bird_direction = np.array([1, 0])

x = np.linspace(-10, 10, 101)
y = np.linspace(-10, 10, 101)

X, Y = np.meshgrid(x, y)


# Define coherence function
def coherence(x, y, theta=1):
    # Avoid division by zero
    distance = np.sqrt(x**2 + y**2)
    scalar_product = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.abs(1 / distance) * scalar_product * theta
        result[np.isnan(result)] = 0  # Replace NaNs caused by division by 0
        result[np.isinf(result)] = 0  # Replace infs as well
    return result


# Define logarithmic coherence function
def logarithmic_coherence(x, y, theta=1):
    distance = np.sqrt(x**2 + y**2)
    scalar_product = 1
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(np.abs(1 / distance)) * scalar_product * theta
        result[np.isnan(result)] = 0  # Replace NaNs caused by division by 0
        result[np.isinf(result)] = 0  # Replace infs as well
    return result


Z = coherence(X, Y)
# Z = logarithmic_coherence(X, Y)
print(Z)

fig = plt.figure(figsize=(10, 8))

ax = sns.heatmap(Z, xticklabels=10, yticklabels=10, cmap="viridis")


# Create ticks for every 10th index
x_tick_positions = np.linspace(0, len(x) - 1, 10, dtype=int)
y_tick_positions = np.linspace(0, len(y) - 1, 10, dtype=int)

# Set ticks with labels corresponding to actual distance and scalar_product values
ax.set_xticks(x_tick_positions)
ax.set_xticklabels([f"{x[i]:.1f}" for i in x_tick_positions])

ax.set_yticks(y_tick_positions)
ax.set_yticklabels([f"{y[i]:.1f}" for i in y_tick_positions])


# Find index closest to x=0 and y=0
x0_index = np.argmin(np.abs(x - 0))
y0_index = np.argmin(np.abs(y - 0))

# Add lines at these index positions
ax.axvline(x=x0_index + 0.5, color="black", linewidth=0.5)
ax.axhline(y=y0_index + 0.5, color="black", linewidth=0.5)

plt.title("Simple Coherence Heatmap around Epicenter")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
