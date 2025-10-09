import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

distances = np.linspace(0, 2, 100)  # x-Axis
scalar_products = np.linspace(-1, 1, 100)  # y-Axis


X, Y = np.meshgrid(distances, scalar_products)


# Define coherence function
def coherence(distance, scalar_product, theta=1):
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.abs(1 / distance) * scalar_product * theta
        result[np.isnan(result)] = 0  # Replace NaNs caused by division by 0
        result[np.isinf(result)] = 0  # Replace infs as well
    return result


Z = coherence(X, Y)

fig = plt.figure(figsize=(8, 8))

ax = sns.heatmap(
    Z,
    xticklabels=10,
    yticklabels=10,
    cmap="viridis",
)


# Create ticks for every 10th index
x_tick_positions = np.linspace(0, len(distances) - 1, 10, dtype=int)
y_tick_positions = np.linspace(0, len(scalar_products) - 1, 10, dtype=int)

# Set ticks with labels corresponding to actual distance and scalar_product values
ax.set_xticks(x_tick_positions)
ax.set_xticklabels([f"{distances[i]:.1f}" for i in x_tick_positions])

ax.set_yticks(y_tick_positions)
ax.set_yticklabels([f"{scalar_products[i]:.2f}" for i in y_tick_positions])

# Find index closest to 0 in distances and scalar_products
x0_index = np.argmin(np.abs(distances - 0))
y0_index = np.argmin(np.abs(scalar_products - 0))

# Add vertical and horizontal lines at index positions
ax.axvline(x=x0_index, color="black", linewidth=0.5)
ax.axhline(y=y0_index, color="black", linewidth=0.5)

plt.title("Coherence Heatmap")
plt.xlabel("Distance")
plt.ylabel("Scalar Product")

plt.show()
