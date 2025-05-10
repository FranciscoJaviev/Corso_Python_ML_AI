'''Nome Script: Analisi sulle malattie da inquinamento del suolo tramite la LinearRegression.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di predire l'insorgenza di malattie
dovute all'inquinamento in sogetti adulti.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [4 maggio 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


df = pd.read_csv('student_habits_performance.csv')



#inizializzo i miei dati
X = df[['exam_score']]
y=df[['study_hours_per_day']]


#Divisione dei Dati in Training e Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Creazione e Addestramento del Modello SVC
#Puoi sperimentare con diversi kernel ('linear', 'poly', 'rbf', 'sigmoid') e parametri
modello_svr = SVR(kernel='linear', C=1.0, gamma='scale')
#modello_svr.fit(X_train, y_train)
modello_svr.fit(X_train, y_train.values.ravel())  #Assicura che y sia un array monodimensionale

# Previsioni sul Set di Test
y_pred = modello_svr.predict(X_test)


#Valutazione
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Error {mae}")

# Creazione del grafico
plt.figure(figsize=(8, 5))

# Dati reali
plt.scatter(X_test, y_test, color='blue', label="Dati Reali", alpha=0.6)

# Previsioni del modello
plt.plot(X_test, y_pred, color='red', label="Previsione SVR", linewidth=2)

# Etichette e titolo
plt.xlabel("Punteggio d'esame")
plt.ylabel("Ore di studio per giorno")
plt.title("Regressione SVR")
plt.legend()
plt.grid(True)
plt.show()
