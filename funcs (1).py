import os 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

regDir = '/home/arthur/main/analysis/data/'
def pathList(directory=regDir) -> list:
    arr = []
    for entry in os.scandir(directory):
        arr.append(entry.path.replace(directory, ''))
    return arr # retorna lista com os chamados numIds (numeros em cada subpasta)

loadXrds : Callable[[str], xr.Dataset]= lambda numberId : xr.open_dataset(regDir + numberId + f'/{numberId}_prof.nc') # carrega o xrds _prof de cada diretorio com numberId especificado

def iter_and_returnXrds(numberIdArr : list) -> list[xr.DataArray]:
    xrdsList = []
    for numberId in numberIdArr:
        xrdsList.append(loadXrds(numberId))

    return xrdsList


def createArr(setDims : list, xrSet : xr.DataArray ,freeDim : str) -> list[tuple[np.float32]]: # versão infinitamente melhor da outra função...
    arr = []
    dim1, dim2, dim3, dim4, dim5 = setDims[0], setDims[1], setDims[2], setDims[3], setDims[4]

    vals1 = xrSet[dim1].dropna(freeDim).values # lista de valores da variavel dim1 depois do filtro NaN. torna mais legível o código, também
    vals2 = xrSet[dim2].dropna(freeDim).values # idem, mas para dim2
    vals3 = xrSet[dim3].dropna(freeDim).values
    vals4 = xrSet[dim4].dropna(freeDim).values
    vals5 = xrSet[dim5].dropna(freeDim).values


    vals1 = np.array(vals1).flatten()
    vals2 = np.array(vals2).flatten() # garante 1 dim, pra que depois possa fazer o slicing
    vals3 = np.array(vals3).flatten()
    vals4 = np.array(vals4).flatten()
    vals5 = np.array(vals5).flatten()

    min_len = min(len(vals1), len(vals2), len(vals3), len(vals4), len(vals5)) # tamanho minimo. isso é necessário pq nem toda tupla é bem comportada, e algumas tem shapes não-homogêneos
    vals1 = vals1[:min_len] # slicing pra garantir mesmo tamanho
    vals2 = vals2[:min_len]
    vals3 = vals3[:min_len]
    vals4 = vals4[:min_len]
    vals5 = vals5[:min_len]

    for i, j, k, q, p in zip(vals1, vals2, vals3, vals4, vals5):
        arr.append( ( float(i), float(j), float(k), float(q), float(p) ) ) # criação da lista de tuplas

    return arr


def concatenateTupleList(xrdsList : list, dataVar1 : str, dataVar2 : str, dataVar3 : str, dataVar4 : str, dataVar5 : str,coord: str) -> list[tuple[np.float32]]:
    tupMaster = []
    for xrds in xrdsList: # em cada xrds, criar uma lista de tuplas
        tup = createArr([dataVar1, dataVar2, dataVar3, dataVar4, dataVar5], xrds, coord)
        tupMaster.extend(tup)
    tupMaster = [i for i in tupMaster if type(i) != float] # filtra e evita que apareça algum float no meio do que deveriam ser tuplas
    return tupMaster # devolve a lista mestra no formato (x,y)


