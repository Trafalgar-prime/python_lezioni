# PyTorch: Output Layer e Loss per Tipo di Task

## Tabella di riferimento rapido

  -----------------------------------------------------------------------
  Tipo di Task            Descrizione             Esempio fisico
  ----------------------- ----------------------- -----------------------
  **Regressione 📈**      Prevedere un valore     Calcolare la
                          numerico continuo.      temperatura di una
                                                  stella.

  **Classificazione       Scegliere tra due       Decidere se un segnale
  Binaria ⚖️**            opzioni mutualmente     è "Rumore" o "Evento".
                          esclusive.              

  **Classificazione       Scegliere **una sola**  Identificare un tipo di
  Multiclasse 🏷️**        opzione tra 3 o più     particella tra 5
                          categorie.              possibili.

  **Classificazione       Assegnare **più         Un materiale può essere
  Multilabel 📑**         etichette               "Magnetico" **e**
                          contemporaneamente**.   "Conduttore".
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 1) Regressione (valori continui)

Qui non classifichiamo: stimiamo una grandezza numerica.

-   **Ultimo layer**: `nn.Linear(n, 1)` (un solo neurone in uscita)
-   **Attivazione**: nessuna (identità) oppure **ReLU** se il valore
    deve essere solo positivo
-   **Loss**:
    -   `nn.MSELoss()` (Mean Squared Error)
    -   `nn.L1Loss()` (Mean Absolute Error)

------------------------------------------------------------------------

# 2) Classificazione binaria

-   **Ultimo layer**: `nn.Linear(n, 1)`
-   **Attivazione**: `torch.sigmoid()` (schiaccia l'output tra 0 e 1 →
    interpretazione probabilistica)
-   **Loss**: `nn.BCELoss()` (Binary Cross Entropy)

------------------------------------------------------------------------

# 3) Classificazione multiclasse (single-label)

Il modello deve scegliere **una sola classe** tra **num_classi**
possibili.

Qui entra in gioco il concetto di: - **One-Hot Encoding** (target come
vettore con un solo "1" e il resto "0") oppure - **indici interi**
(target come intero 0..num_classi-1)

-   **Ultimo layer**: `nn.Linear(n, num_classi)`
-   **Attivazione**: `torch.softmax()` per ottenere una distribuzione di
    probabilità (somma = 1)
-   **Loss**: `nn.CrossEntropyLoss()`

### Nota tecnica (importantissima)

In PyTorch, `nn.CrossEntropyLoss()` **include già** internamente
`LogSoftmax` + `NLLLoss`.\
Quindi l'ultimo layer del modello dovrebbe essere **lineare** (senza
softmax nel `forward`).

------------------------------------------------------------------------

# 4) Classificazione multilabel

Ogni etichetta è trattata come un problema binario indipendente (Sì/No
per ciascuna etichetta).

-   **Ultimo layer**: `nn.Linear(n, num_labels)`
-   **Attivazione**: `torch.sigmoid()` applicata a ogni neurone in
    uscita
-   **Loss**: `nn.BCEWithLogitsLoss()`

### Nota tecnica

`BCEWithLogitsLoss()` integra internamente la sigmoid in modo
numericamente stabile.\
In questo caso, l'ultimo layer del modello dovrebbe rimanere **lineare**
(logits).

------------------------------------------------------------------------

# Un dubbio sulla Multiclasse 🧠

Nella classificazione multiclasse (es. distinguere tra 3 tipi di
galassie), il modello deve fornire **un'unica risposta**:\
cioè **una sola classe** scelta tra tutte quelle possibili.
