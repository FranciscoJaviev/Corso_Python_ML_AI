'''
Nome Script: Secondo script di MachineLearning.py
Descrizione: Simuliamo un piccolo dataset di studenti, con ore di studio, ore di sonno, e voto finale.  
Calcolando la media, la varianza e la deviazione standard
- Alleniamo un modello per vedere se le ore di studio/sonno potrà prevedere il voto. Successivamente
creiamo un file che sarà ingrato di fare previsioni con nuovi dati, tramite il modello che ha
imparato tramite questo script.

Autore: [Francisco J.Scognamiglio]
Versione: 1.0
Data: [15.4.25]
Copyright: © [Francisco J.Scognamiglio] [2025]
Licenza: [MIT]
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump



#Creiamo un piccolo dataset
data = {
    "ore_studio": [2, 4, 6, 8, 10],
    "ore_sonno":  [ 8, 7, 6, 5, 4],
    "voto_finale":   [60, 65, 70, 75, 85]
}
df = pd.DataFrame(data)

print("Media:\n", df.mean(numeric_only=True))
print("Varianza:\n", df.var(numeric_only=True))
print("Deviazione Standard:\n", df.std(numeric_only=True))


#Separiamo le variabili
X = df[["ore_studio", "ore_sonno"]] #variabili indipendenti
y = df["voto_finale"] #variabili dipendenti


#Suddividiamo in training e in test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Modello di ML: Regressione Lineare
model =  LinearRegression()
model.fit(X_train, y_train)

y_predizione = model.predict(X_test)


#Valutazione
print(f"voti predetti: {y_predizione}")
print(f"voti reali: {y_test}")

dump(model, "modello_regressione_lineare.pkl")