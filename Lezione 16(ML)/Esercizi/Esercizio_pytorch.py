import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# il train split e la normalizzazione dei dati avviene sempre sklearn

dati = load_digits()
X = dati.data
Y = dati.target

print("X : ",X,"\n")
print("Y : ",Y,"\n")
print("Forma di X : ",X.shape,"\n")
print("Forma di Y : ",Y.shape,"\n")
print(len(np.unique(Y)),"\n") 
print(np.unique(Y),"\n")   

X_train, X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

print("\n")
print("Dimensioni della X_train:\n",X_train.shape)
print("\n")
print("Dimensioni della Y_train:\n",Y_train.shape)
print("\n")
print("Dimensioni della X_test:\n",X_test.shape)
print("\n")
print("Dimensioni della Y_test:\n",Y_test.shape)
print("\n")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

