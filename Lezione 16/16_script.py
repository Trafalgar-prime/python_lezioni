import pandas as pd
from sklearn.datasets import load_iris  # Importa il dataset Iris

# Caricare un dataset (Esempio: Iris dataset)
dati = load_iris()  # Carica i dati di Iris

# Creiamo un DataFrame Pandas con i dati e aggiungiamo la colonna "target"
df = pd.DataFrame(dati.data, columns=dati.feature_names)
df['target'] = dati.target

# Mostriamo le prime 5 righe del dataset
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# Dividiamo il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)

# Normalizziamo i dati per migliorare l'addestramento
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creiamo una rete neurale con un solo livello nascosto di 10 neuroni
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)  # Addestriamo la rete

# Valutiamo il modello
accuracy = mlp.score(X_test, y_test)
print(f"Accuratezza: {accuracy:.2f}")

# Predizioni sul test set
y_pred = mlp.predict(X_test)

# Creazione della confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
