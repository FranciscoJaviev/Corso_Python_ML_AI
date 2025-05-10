'''Nome Script: Analisi sulla depressione negli adulti tramite la LinearRegression.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di predire l'insorgenza della
depressione nei sogetti adulti.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [1 maggio 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''
#import numpy as np
#from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



df = pd.read_csv('adult-depression-lghc-indicator-24.csv')

#elimino tutte le colonne che non contengono numeri
df_1 = df.drop(df.select_dtypes(include=['object']).columns, axis=1)


#visualizzo le correlazioni che ci sono tra le diverse colonne del dataset
fig, ax = plt.subplots(figsize=(12, 8))
corr = df_1.corr()
sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0, annot=True,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
    
)
plt.show()


'''

X = df[['Upper 95% CL']]
y=df[['Lower 95% CL']]

#Divisione dei Dati in Training e Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Regressione Lineare tramite un polinomio
#Specifica il grado del polinomio che vuoi adattare
grado_polinomio = 1

# Crea un oggetto PolynomialFeatures
polinomio = PolynomialFeatures(degree=grado_polinomio)

# Trasformazione delle variabili indipendenti
X_train_poly = polinomio.fit_transform(X_train)
X_test_poly = polinomio.transform(X_test)

# Creazione e addestramento del modello
modello_polinomiale = LinearRegression()
modello_polinomiale.fit(X_train_poly, y_train)

# Puoi ottenere i coefficienti del modello polinomiale
print("Coefficienti del modello polinomiale:", modello_polinomiale.coef_)
print("Intercetta del modello polinomiale:", modello_polinomiale.intercept_)

y_pred = modello_polinomiale.predict(X_test_poly)
'''
'''
#Valutazione
mse = mean_squared_error(df['Frequency'].values, y_pred)
r2 = r2_score(df['Frequency'].values, y_pred)

print(f"Mean Squared Error (con 2 variabili): {mse:.2f}")
print(f"R^2 Score (con 2 variabili): {r2:.2f}")
'''

#Creo il grafico per visualizzare i dati originali e polinomio 
'''
plt.figure(figsize=(6, 5))


# Dati reali
plt.scatter(X_test, y_test, color='blue', label="Dati Reali", alpha=0.6)

# Previsioni del modello
plt.plot(X_test, y_pred, color='red', label="Previsione SVR", linewidth=2)


plt.xlabel('Upper 95% CL')
plt.ylabel('Lower 95% CL')
plt.title('Regressione Polinomiale')
plt.legend()
plt.grid(True)

plt.show()
'''
