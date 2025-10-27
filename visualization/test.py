import numpy as np
from matplotlib import pyplot as plt


def exponential(d, a=1):
    return np.exp(a / d)


numbers = np.linspace(1, 5, 100)

plt.plot(numbers, exponential(numbers))
plt.show()
