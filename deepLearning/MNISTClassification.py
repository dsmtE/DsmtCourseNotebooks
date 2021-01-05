# Huge help : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_CNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.optim.optimizer import Optimizer, required # needed in order to create custom Optimizer
import torch.optim as optim
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import skimage.io as skio # image read and save

import matplotlib.pyplot as plt
import numpy as np
import time

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# save model Path
savePath = './MNISTClassification.pth'

# Hyperparameters
nbClasses = 10
learningRate = 1e-3
batchSize = 200
epochs = 20

def cmAccuracy(cm):
    return np.trace(cm) / float(np.sum(cm))

def cmByClassesAccuracy(cm):
    return np.diag(cm) / np.sum(cm, 1)

def confusionMatrix(data, targets, model, normalize = False):
    print(data.size())
    nbClasses = len(targets.unique())

    inputs, targets = data.to(device=device), targets.to(device=device)

    cm = np.zeros((nbClasses, nbClasses))

    model.eval()
    with torch.no_grad():

        predictions = model(inputs).argmax(1)
        for pred, targ in zip(predictions, targets): 
            cm[targ][pred] += 1

    model.train()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm

def plotConfusionMatrix(cm, target_names, title='Confusion matrix', cmap=None, normalize = True):
    accuracy = cmAccuracy(cm)

    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im, shrink=0.75)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in np.dstack(np.mgrid[0:cm.shape[0], 0:cm.shape[1]]).reshape(-1, 2) :
        plt.text(j, i, f'{cm[i, j]:0.4f}' if normalize else f'{cm[i, j]:,}', horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'accuracy={accuracy:0.4f}')
    plt.show()

# Load Data
trainDataset = datasets.MNIST( root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
testDataset = datasets.MNIST( root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(16*7*7, 50),
    nn.ReLU(),
    nn.Linear(50, nbClasses),
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=learningRate)

loss = nn.CrossEntropyLoss()

trackLoss = True
epochsLoss = []

# loading model
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
try:
    checkpoint = torch.load(savePath)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded")

except:
    print("Model checkpoint not found")
    
# print("\n----- Start training -----\n")

# since = time.time()

# # Train Network
# for epoch in range(epochs):

#     epochLoss = 0
#     for batchIdx, (inputs, targets) in enumerate(trainLoader, 1):
        
#         inputs, targets = inputs.to(device=device), targets.to(device=device) # Get data (to cuda if possible)

#         optimizer.zero_grad() # zero the parameters gradients

#         outputs  = model(inputs) # forward
#         lossValue = loss(outputs, targets)  # compute loss

#         if(trackLoss): # record loss
#             epochLoss += lossValue.item()

#         lossValue.backward() # backward
        
#         optimizer.step() # use custom optimizer class instead of manualy update model parameters

#     if(trackLoss): # record loss
#         epochsLoss.append(epochLoss / len(trainLoader))
    
#     if epoch % (epochs/10) == (epochs/10-1):
#         print(f"Epoch[{epoch:>4}] Complete: Avg. Loss: {epochLoss / len(trainLoader):.8f}")

# time_elapsed = time.time() - since
# print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

# if savePath is not None:
#     torch.save(model.state_dict(), savePath)
#     print('model saved ')


targetNames=testDataset.targets.unique().numpy().astype('U')
plotConfusionMatrix(confusionMatrix(trainDataset.data.float().unsqueeze(1), trainDataset.targets , model), target_names=targetNames, title='train dataset confusion matrix')
plotConfusionMatrix(confusionMatrix(testDataset.data.float().unsqueeze(1), testDataset.targets , model), target_names=targetNames, title='test dataset confusion matrix')

if(trackLoss): # display recored loss values
    plt.plot(epochsLoss, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
