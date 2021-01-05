import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# import torchvision.transforms as transforms # Transformations we can perform on our dataset
from torch.optim.optimizer import Optimizer, required # needed in order to create custom Optimizer
import torch.optim as optim

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader # Gives easier dataset managment for batches

from torchvision.utils import save_image

from math import log, floor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time

# image mesurment
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# save model Path
savePath = './MNIST_DAE.pth'

# Hyperparameters
learningRate = 1e-3
batchSize = 200
epochs = 30

if not os.path.exists('./MNIST_DAE_Report'):
    os.mkdir('./MNIST_DAE_Report')

def makeGrid(array, nrows=3):
    nindex, height, width = array.shape
    ncols = nindex//nrows
    assert nindex == nrows*ncols
    return array.reshape(nrows, ncols, height, width, -1).swapaxes(1,2).reshape(height*nrows, width*ncols, -1)

def displayRowComparaison(data, dataDescription = None, labels = None, labelsAnnotations = None, cleanAxis = True, hAxesPad=0., vAxesPad = 0.):
    rows, cols = len(data), len(data[0])

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

def noise(size, alpha = 0.5):
    return alpha * torch.randn(size)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
            
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load Data
trainDataset = MNIST( root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)

testDataset = MNIST( root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=True)

model = Autoencoder().to(device)

optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

loss = nn.BCELoss()

trackLoss = True
epochsLoss = []

# loading model
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
try:
    checkpoint = torch.load(savePath)
    model.load_state_dict(checkpoint)
    print("Model loaded")

except:
    print("Model checkpoint not found")

def train(epochs):
    since = time.time()
    epochPadding = floor(log(epochs))

    for epoch in range(epochs):

        epochLoss = 0
        for batchIdx, (inputs, _) in enumerate(trainLoader, 1):
            inputs = inputs.to(device=device) # Get data (to cuda if possible)

            inputs = torch.flatten(inputs, start_dim=1)
            noisyInputs = inputs +  noise(inputs.size()).to(device=device)
            
            outputs = model(noisyInputs) # forward
            lossValue = loss(outputs, inputs)  # compute loss
            if(trackLoss): # record loss
                epochLoss += lossValue.item()

            optimizer.zero_grad() # zero the parameters gradients
            lossValue.backward() # backward
            optimizer.step()

        if(trackLoss): # record loss
            epochsLoss.append(epochLoss / len(trainLoader))
        
        if epoch % (epochs/10) == (epochs/10-1):
            print(f"Epoch[{epoch+1:>{epochPadding}}/{epochs}] Complete: Avg. Loss: {epochLoss / len(trainLoader):.8f}")

    return time.time() - since

print("\n----- Start training -----\n")
timeElapsed = train(epochs)
print(f'Training complete in {timeElapsed // 60:.0f}m {timeElapsed % 60:.0f}s')

if savePath is not None:
    torch.save(model.state_dict(), savePath)
    print('model saved')


data = testDataset.data.numpy().astype(np.float32) / 255.0
targets = testDataset.targets.numpy()

# subsample
data = data[:1000]
targets = targets[:1000]

# shuffle the same way
p = np.random.permutation(len(data))
data = data[p]
targets = targets[p]

cols = 10
# subsample
data = data[:cols]
targets = targets[:cols]

# compute result
dataTensor = torch.flatten(torch.tensor(data), start_dim=1)
noisyData = dataTensor + noise(dataTensor.size())

model.eval()
with torch.no_grad():
    denoised = model(noisyData.to(device=device))
model.train()

# unnormalize and convert back tensors to numpy image arrays uint8
# Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
denoised = denoised.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).view(denoised.size(0), 28, 28).numpy()
noisyData = noisyData.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).view(noisyData.size(0), 28, 28).numpy()
data = dataTensor.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).view(dataTensor.size(0), 28, 28).numpy()

psnr = [round(psnr(a, b), 2) for a, b in zip(data, denoised)]
mse = [round(mean_squared_error(a, b), 2) for a, b in zip(data, denoised)]

fig = displayRowComparaison(np.stack((data, noisyData, denoised)), ['data', 'noisyData', 'denoised'], targets.astype('U'), [f'psnr: {p}\nmse: {m}' for p, m in zip(psnr, mse)])
fig.savefig('./MNIST_DAEReport.png')

if(trackLoss): # display recored loss values
    plt.plot(epochsLoss, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()