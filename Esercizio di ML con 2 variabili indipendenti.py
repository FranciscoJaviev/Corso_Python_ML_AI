'''Nome Script:  Esecitazione fatto in classe utilizznado il MSE e R2_score.
Descrizione: Questo script elabora e analizza i dati di di un database fatto in classe utilizznado 
il MSE e R2_score, dove si cerca di predire il punteggio tramite due variabili indipendenti.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [22 aprile 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#altro modo
'''
ore_studio_sonno = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])  
[10, 9, 9, 10, 8, 7, 8, 6, 10, 9, 9, 10, 8, 7, 6, 8]])

# Suddivisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(ore, punteggi, test_size=0.2, random_state=42)

# Creazione e addestramento del modello
modello = LinearRegression()
modello.fit(ore, punteggi)

'''

#variabili dipendenti
ore_studio = np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8])  
ore_sonno = np.array([10, 9, 9, 10, 8, 7, 8, 6, 10, 9, 9, 10, 8, 7, 6, 8])

#varibile indipendente
punteggi = np.array([48, 52, 60, 66, 70, 77, 92, 100, 48, 52, 60, 66, 70, 77, 92, 100 ])

# Creiamo una matrice con entrambe le variabili indipendenti (features)
X = np.column_stack((ore_studio, ore_sonno))

# Suddivisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, punteggi, test_size=0.2, random_state=42)

# Creazione e addestramento del modello
modello = LinearRegression()
modello.fit(X, punteggi)

# Fare previsioni sui dati di test
y_pred = modello.predict(X_test)

# Valutare il modello
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualizzare i risultati
print(f"Mean Squared Error (Senza outlier): {mse}")
print(f"R^2 Score (Senza outlier): {r2}")


#creazione figura in 3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#Scatter plot dei punti dati
ax.scatter(ore_studio, ore_sonno, punteggi, color='red', label='Dati reali')

# Creazione della superficie del modello di regressione
xx, yy = np.meshgrid(np.linspace(min(ore_studio), max(ore_studio), 10),
                     np.linspace(min(ore_sonno), max(ore_sonno), 10))
zz = modello.predict(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

# Disegna la superficie
ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue')

# Etichette
ax.set_xlabel("Ore di Studio")
ax.set_ylabel("Qualità del Sonno")
ax.set_zlabel("Punteggio Esame")
ax.legend()

plt.show()
