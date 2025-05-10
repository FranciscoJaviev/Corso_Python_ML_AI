'''Nome Script:  Analisi sul costo delle abbitazioni in USA tramite la LinearRegression.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di predire il costo della casa in
USA in base a dei parametri scelti.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [1 maggio 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('USA Housing Dataset.csv')
'''
#elimino tutte le colonne che non contengono numeri
df_1 = df.drop(df.select_dtypes(include=['object']).columns, axis=1)

#visualizzo le correlazioni che ci sono tra le diverse colonne del dataset
fig, ax = plt.subplots(figsize=(8, 5))
corr = df_1.corr()
sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0, annot=True,
    cmap=sns.dark_palette("yellow", 20, True),
    square=True
    
)
'''

X = df[['bedrooms', 'sqft_living']]
y = df['price'] 

#Regressione Lineare tramite un polinomio
#Specifica il grado del polinomio che vuoi adattare
grado_polinomio = 1

# Crea un oggetto PolynomialFeatures
polinomio = PolynomialFeatures(degree=grado_polinomio)

# Trasforma le variabili indipendenti originali in termini polinomiali
# Trasforma le variabili indipendenti in termini polinomiali
X_poly = polinomio.fit_transform(X)

# Ora puoi usare LinearRegression sui termini polinomiali trasformati
modello_polinomiale = LinearRegression()
modello_polinomiale.fit(X_poly, y)

# Puoi ottenere i coefficienti del modello polinomiale
print("Coefficienti del modello polinomiale:", modello_polinomiale.coef_)
print("Intercetta del modello polinomiale:", modello_polinomiale.intercept_)


#Ottiengo le previsioni dal modello
y_pred = modello_polinomiale.predict(X_poly)

#Valutazione
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (con 3 variabili): {mse:.2f}")
print(f"R^2 Score (con 3 variabili): {r2:.2f}")

#Creo il grafico per visualizzare i dati originali e polinomio 

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

#Plot dei dati originali

scatter = ax.scatter(df['bedrooms'], df['sqft_living'], df['price'], c='blue', marker='o', label='Dati Reali')

#plot dei valori predetti
scatter_pred = ax.scatter(df['bedrooms'], df['sqft_living'], y_pred, c='red', marker='x', label='Valori Predetti')


ax.set_xlabel('Numero di Camere da Letto')
ax.set_ylabel('Metri Quadrati di Superficie Abitabile')
ax.set_zlabel('Prezzo')

ax.legend()
plt.title('Regressione Polinomiale: Dati Reali vs. Valori Predetti')

#plt.show()
