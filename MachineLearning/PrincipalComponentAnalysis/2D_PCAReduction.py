from scipy.linalg import eigh

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sn

import numpy as np
import time
import math


rng = np.random.RandomState(1)
data = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

# compute cov matrix
# cm = np.matmul(standardizedData.T, standardizedData) / (standardizedData.shape[0]-1)
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

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data and eigenVector
plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
for length, vector in zip(varianceRatio, vectors):
    v = vector * np.sqrt(length)
    draw_vector(np.mean(data, axis=0), np.mean(data, axis=0) + v)
plt.axis('equal')
plt.show()

principalEv = vectors[-1, :][:, np.newaxis]
print(f'principalEv:\n {principalEv}')


projected = np.dot(data, principalEv)
print(f'projected.shape:{projected.shape}')
backProjected = np.matmul(projected, principalEv.T)
print(f'backProjected.shape:{backProjected.shape}')

plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
for length, vector in zip(varianceRatio, vectors):
    v = vector * np.sqrt(length)
    draw_vector(np.mean(data, axis=0), np.mean(data, axis=0) + v)
plt.scatter(backProjected[:, 0], backProjected[:, 1])
plt.axis('equal')
plt.show()
