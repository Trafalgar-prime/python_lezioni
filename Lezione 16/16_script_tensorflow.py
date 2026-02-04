import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  # Importa il dataset Iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras


# Caricare un dataset (Esempio: Iris dataset)
dati = load_iris()  # Carica i dati di Iris
#Questa funzione carica in memoria il dataset Iris e lo restituisce come un oggetto “contenitore” di scikit-learn.

# Creiamo un DataFrame Pandas con i dati e aggiungiamo la colonna "target"
df = pd.DataFrame(dati.data, columns=dati.feature_names)
#pd.set_option('display.max_rows', None) #serve per vedere tutti i dati e non solo i parziali in output, perche altrimenti pandas mi stampa solo i parziali per non intasare l'output
#pd.set_option('display.max_columns', None) #stesso del precedente ma per le colonne [in questo caso è inutile perche le colonne sono solo 4]
print(df)
print("\nProva\n")
print(df.shape)
print(df.columns)
print("\n")


df.index.name = "id" # la colonna degli indici la stampa pandas in automatico, in questo modo gli do un nome
df['target'] = dati.target


#Mostriamo le prime 5 righe del dataset
print(df.head())
print("\n",df,"\n")
print("\n")


# Dividiamo il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)
#.iloc permette di definire le posizioni di quello di cui ho bisogno in base alle colonne

# Normalizziamo i dati per migliorare l'addestramento
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



modello = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),  # 16 neuroni
    keras.layers.Dense(8, activation='relu'),  # 8 neuroni
    keras.layers.Dense(3, activation='softmax')  # 3 classi in output
])

"""
Cos’è Sequential
Sequential significa che la rete è una pipeline lineare:
output layer 1 → input layer 2
output layer 2 → input layer 3
ecc.
Quindi nessuna ramificazione, nessun “salto” (skip connection). Per Iris va benissimo.
"""

modello.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modello.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
