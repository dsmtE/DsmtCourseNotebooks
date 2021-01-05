from scipy.linalg import eigh

# use torchvision to load MNIST dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# image mesurment
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import time
import math

np.set_printoptions(suppress = True)

def makeGrid(array, nrows=3):
    nindex, height, width = array.shape
    ncols = nindex//nrows
    assert nindex == nrows*ncols
    return array.reshape(nrows, ncols, height, width, -1).swapaxes(1,2).reshape(height*nrows, width*ncols, -1)

def displayRowComparaison(data, dataDescription = None, labels = None, labelsAnnotations = None, cleanAxis = True, hAxesPad=0., vAxesPad = 0.):
    rows, cols = len(data), len(data[0]) # data.shape[:2]

    for i in range(1, rows):
        assert data[i-1].shape == data[i].shape, 'all data must have the same shape'

    assert len(dataDescription) is None or rows == len(dataDescription), 'dataDescription must match the number of data'
    assert len(labels) is None or cols == len(labels), 'labels must match the number of labels in data'
    assert len(labels) == len(labelsAnnotations), 'labels size must match labelsAnnotations size' 

    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2+1), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=vAxesPad, wspace=hAxesPad)

    for ax, img in zip(axs.ravel(), data.reshape(-1, *data.shape[-2:])): # display images
        ax.imshow(img, cmap='gray')
    
    for ax, l in zip(axs[0], labels): # add target titles
        ax.set_title(l)
    
    if cleanAxis: # clean axis
        for ax in axs.ravel():
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    for ax, desc in zip(axs[:, 0], dataDescription):
        ax.get_yaxis().set_visible(True)
        if cleanAxis:
            ax.set_yticklabels([])
        ax.set_ylabel(desc, rotation=90, size='large')

    for ax, annotation in zip(axs[-1], labelsAnnotations):
        ax.get_xaxis().set_visible(True)
        if cleanAxis:
            ax.set_xticklabels([])
        ax.set_xlabel(annotation)
    plt.show()
    return fig

MNISTDataset = datasets.MNIST( root="dataset/", train=True, transform=transforms.ToTensor(), download=True)

data = MNISTDataset.data.numpy()
data = data.reshape(data.shape[0], -1)
targets = MNISTDataset.targets.numpy()

# # subsample
# data = data[:1000]
# targets = targets[:1000]

# shuffle the same way
p = np.random.permutation(len(data))
data = data[p]
targets = targets[p]

print(f"data{data.shape}: dtype: {data.dtype}")
print(f"targets{targets.shape}: dtype: {targets.dtype}")

# displayGrid(2, 4, data[:8].reshape(-1, 28, 28), targets[:8].astype('U'))

# Data-preprocessing: Standardizing the data
# # It is mandatory before applying PCA to convert mean = 0 and standard deviation = 1 for each variable.
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
#handle zero in scale (all values the same)
std[std == 0.0] = 1.0
standardizedData = (data - mean) / std
#compute cov matrix
# cm = np.matmul(standardizedData.T, standardizedData) / (standardizedData.shape[0]-1)
cm = np.cov(standardizedData, rowvar=False)

# finding eigen-values and corresponding eigen-vectors 
values, vectors = eigh(cm)

print(f'eigen-vectors:\n {np.mean(vectors**2, axis=0)}')
# np.savetxt('./vectors.csv', vectors, delimiter=';', fmt='%1.1f')

# print(f'eigen-values:\n {values}')
print(f'eigen-values:\n {values / np.sum(values)}')
# varianceRatio = values / np.sum(values)
# print(f'varianceRatio:\n {varianceRatio}')

# plot
explainedVarianceSum = np.cumsum(values[::-1] / np.sum(values))
# plt.plot(explainedVarianceSum)
# plt.show()

Variance99Idx = np.argwhere(explainedVarianceSum > 0.90).min()
print(f'Variance99Idx: {Variance99Idx}')

principalEv = vectors[:, -Variance99Idx:].T
print(f'principalEv{principalEv.shape}')

# shuffle and subsample
p = np.random.permutation(len(data))
# subData = standardizedData[:1000][p][:10]
# subTargets = targets[:1000][p][:10]
subData = standardizedData[p]
subTargets = targets[p]

projected = subData @ principalEv.T
backProjected = projected @ principalEv

# projected = np.matmul(principalEv, subData.T).T
# backProjected = np.matmul(principalEv.T, projected.T).T

# W = vectors[:,-3:]  # just three dimensions
# proj_digits = subData @ W
# print("proj_digits.shape = ", proj_digits.shape)

# # Make the plot, separate them by "z" which is the digit of interest.  
# fig = go.Figure(data=[go.Scatter3d(x=proj_digits[:,0], y=proj_digits[:,1], z=proj_digits[:,2],
#                 mode='markers', marker=dict(size=4, opacity=0.8, color=subTargets, showscale=True), 
#                 text=['digit='+str(j) for j in subTargets] )])
# fig.update_layout(title="8x8 Handwritten Digits", xaxis_title="q_1", yaxis_title="q_2", yaxis = dict(scaleanchor = "x",scaleratio = 1) )
# fig.show()

subData = np.clip(subData * std + mean, 0, 255).astype(np.uint8)[:8]
backProjected = np.clip(backProjected * std + mean, 0, 255).astype(np.uint8)[:8] 
subTargets = subTargets[:8]

print(f'subData{subData.shape}: {subData.dtype}')
# print(f'projected{projected.shape}: {projected.dtype}')
print(f'backProjected{backProjected.shape}: {backProjected.dtype}')

psnr = [round(psnr(a, b), 2) for a, b in zip(subData, backProjected)]
mse = [round(mean_squared_error(a, b), 2) for a, b in zip(subData, backProjected)]

displayRowComparaison(np.stack((subData.reshape(-1, 28, 28), backProjected.reshape(-1, 28, 28))), ['data', 'backProjected'],  subTargets.astype('U'), [f'psnr: {p}\nmse: {m}' for p, m in zip(psnr, mse)])
