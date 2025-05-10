'''
Nome Script: Primo script di MachineLearning.py
Descrizione: Questo script elabora attraverso il ML di classificazione, la previsione di un
esame attraverso i dati, impara da essi e crea il modello con il quale potremo fare previsioni.
Genera successivamente un file che ha il modello previsionale addestrato.

Autore: [Francisco J.Scognamiglio]
Versione: 1.0
Data: [14.4.25]
Copyright: © [Francisco J.Scognamiglio] [2025]
Licenza: [MIT]
'''

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump


#Creiamo un piccolo dataset
data = {
    "ore_studio": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "ore_sonno":  [8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 2, 2, 5, 7, 7, 6, 6, 5, 5, 4, 4, 3, 8, 8],
    "superato":   [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)



#Statistiche descrittive
print("Media:\n", df.mean(numeric_only=True))
print("\nMediana:\n", df.median(numeric_only=True))
print("\nModa:\n", df.mode(numeric_only=True).iloc[0])


#Separiamo le variabili
X = df[["ore_studio", "ore_sonno"]]
y = df["superato"]


#Suddividiamo in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Modello di ML: Regressione Logistica
model = LogisticRegression()
model.fit(X_train, y_train)


#Valutazione
y_predizione = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_predizione))

dump(model, "modello_classificazione.pkl")