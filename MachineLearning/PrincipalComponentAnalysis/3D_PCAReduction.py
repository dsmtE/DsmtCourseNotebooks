from scipy.linalg import eigh

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sn

import numpy as np
import time
import math

rng = np.random.RandomState(1)
data = np.dot(rng.rand(3, 3), rng.randn(3, 200)).T

# compute cov matrix
cm = np.cov(data, rowvar=False)
print(f'cov matrix:\n {cm}')

# finding eigen-values and corresponding eigen-vectors 
values, vectors = eigh(cm)

print(f'eigen-vectors:\n {vectors}')
print(f'eigen-values:\n {values}')
varianceRatio = values / np.sum(values)
print(f'varianceRatio:\n {varianceRatio}')

# plot 
# plt.plot(np.cumsum(values[::-1] / np.sum(values)))
# plt.show()

principalEv = vectors[:, -2:]
print(f'principalEv:\n {principalEv}')


projected = np.dot(data, principalEv)
print(f'projected.shape:{projected.shape}')
backProjected = np.matmul(projected, principalEv.T)
print(f'backProjected.shape:{backProjected.shape}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.scatter(backProjected[:, 0], backProjected[:, 1], backProjected[:, 2])
plt.show()

