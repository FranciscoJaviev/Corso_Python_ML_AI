'''Nome Script:  Analisi sul tumore al seno tramite la LinearRegression.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di
predire l'andamento tra due varibili maggiormente correlate.

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
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Breast Cancer Wisconsin (Diagnostic).csv')

#elimino tutte le colonne che non contengono numeri
df_1 = df.drop(df.select_dtypes(include=['object']).columns, axis=1)  
'''
#elimino una serie di colonne
df_1 = df.drop(df.columns[1:20], axis=1)

#grafico con le correlazioni tra le restanti colonne numeriche
fig, ax = plt.subplots(figsize=(8, 6))
corr = df_1.corr()

sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0, annot=True,
    cmap="coolwarm",
    square=True
    
)
'''


# Calcolo della correlazione tra le variabili X e y
df1 = df[['perimeter_worst', 'radius_worst']]
corr = df1.corr()
print(corr)

correlazione1 = df['perimeter_worst'].corr(df['radius_worst'], method='spearman')
correlazione2 = df['perimeter_worst'].corr(df['radius_worst'], method='kendall')
print("Correlazione tra perimetro peggiore e raggio peggiore (spearman):\n", correlazione1)
print("Correlazione tra perimetro peggiore e raggio peggiore (kendall):\n", correlazione2)


#Regressione Lineare tramite un polinomio
#Specifica il grado del polinomio che vuoi adattare
grado_polinomio = 1

# Crea un oggetto PolynomialFeatures
polinomio = PolynomialFeatures(degree=grado_polinomio)

# Trasforma le variabili indipendenti originali in termini polinomiali
X_poly = polinomio.fit_transform(df['perimeter_worst'].values.reshape(-1, 1))

# Ora puoi usare LinearRegression sui termini polinomiali trasformati
modello_polinomiale = LinearRegression()
modello_polinomiale.fit(X_poly, df['radius_worst'])

# Puoi ottenere i coefficienti del modello polinomiale
print("Coefficienti del modello polinomiale:\n", modello_polinomiale.coef_)
print("Intercetta del modello polinomiale:\n", modello_polinomiale.intercept_)


x_range = np.linspace(df['perimeter_worst'].min(), df['perimeter_worst'].max(), 569).reshape(-1, 1)
X_range_poly = polinomio.transform(x_range)
y_pred = modello_polinomiale.predict(X_range_poly)


#Valutazione
mse = mean_squared_error(df['perimeter_worst'].values, y_pred)
r2 = r2_score(df['perimeter_worst'].values, y_pred)

print(f"Mean Squared Error (con 2 variabili): {mse:.2f}")
print(f"R^2 Score (con 2 variabili): {r2:.2f}")


#Creo il grafico per visualizzare i dati originali e polinomio 


plt.figure(figsize=(6, 5))
#Plot dei dati originali
plt.scatter(df['perimeter_worst'], df['radius_worst'], color='blue', label='Dati Originali')

#Plot del polinomio
plt.plot(x_range, y_pred, color='red', label=f'Regressione Polinomiale (Grado {grado_polinomio})')


plt.xlabel('perimetro peggiore')
plt.ylabel('raggio peggiore')
plt.title('Regressione Polinomiale')
plt.legend()
plt.grid(True)

plt.show()



