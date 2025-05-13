import xarray as xr
import numpy as np
import gsw

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as functional
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Generator


class dataset(Dataset):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.inputs = torch.from_numpy(inputs).float()
        self.outputs = torch.from_numpy(outputs).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def createDataLoader(batch_size, dataset):
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers = 2)

    for batch in dataloader:
        x, y = batch

        print(f"Batch size {x.shape}, labels: {y.shape}")

        break


def kFoldCrossValidation(torchDataArray : Dataset, batchSize : int, nSplits : int, shuffle=True) -> Generator[DataLoader, Dataset, None]: #yielda um DataLoader (iterable) quando enviado um dataset
    kfold = KFold(n_splits = nSplits, shuffle=shuffle)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(torchDataArray)):

        trainSubset = Subset(torchDataArray, train_idx)
        validationSubset = Subset(torchDataArray, val_idx)

        trainLoader = DataLoader(trainSubset, batch_size = batchSize, num_workers = 2) # iterable
        validationLoader = DataLoader(validationSubset, batch_size = batchSize, shuffle=False, num_workers = 2)

        yield trainLoader, validationLoader, trainSubset, validationSubset


def kFoldTrain(model, epochs: int, lossFunction, optim, generatorObject) -> tuple[list[np.float32], list[np.float32]]:
    train_losses = []  # average training loss per epoch
    val_losses = []    # average validation loss per epoch

    best_val_loss = float('inf')
    patience = 10

    for fold, (train_loader, val_loader, _, _) in enumerate(generatorObject):
        epochs_no_improve = 0
        print("---------------------------------------------------------------")
        print(f"Fold: {fold + 1}")

        for epoch in range(epochs):
            counter = 0
            # Training phase
            model.train()
            epoch_train_losses = []

            for batch in train_loader:
                inputs, targets = batch

                optim.zero_grad()
                outputs = model(inputs)
                loss = lossFunction(outputs, targets)
                loss.backward()
                optim.step()
                epoch_train_losses.append(loss.item())

                counter += 1
            print(f"Epoch: {epoch + 1} \t Batches: {counter} \t Loss: {loss}")
            # validation phase
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = lossFunction(outputs, targets)
                    epoch_val_losses.append(loss.item())
            
            # average training loss values for this epoch
            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = np.mean(epoch_val_losses)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # early-stoppage implementation
            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                patience -= 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break



    return train_losses, val_losses

def plotTrainAndValLosses(lossListPlot : list, valLossListPlot : list):
    plt.title("Training and Validation Losses as functions of 'epochs' (KFCV)")
    plt.scatter(np.arange(0, len(lossListPlot)),lossListPlot)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss at the end of fold")
    plt.scatter(np.arange(0, len(valLossListPlot)),valLossListPlot)
    plt.axhline(np.mean(lossListPlot[-1]), ls= '--')
    plt.legend(("$\lim_{j → \infty}\mathcal{L}$ = " + str(round(np.mean(lossListPlot), 2)), "Training Loss", "Validation Loss"))
    return None

def resRed(yArr, o):
    yArr = yArr.flatten()
    o = o.flatten()
    min_len = min(len(yArr), len(o))

    yArr = yArr[:min_len]
    o = o[:min_len]

    res = yArr - o
    plt.title("Residual analysis for the model")
    plt.scatter(o, res)
    plt.axhline(0, color='r', ls='--')
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")

    return 0

