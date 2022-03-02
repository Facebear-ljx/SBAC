import numpy as np
import matplotlib.pyplot as plt

a = 5.  # shape
mu, sigma = 0, 1
s = np.random.normal(mu, sigma, 1000)
x = np.arange(1, 100.) / 50.

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
plt.show()