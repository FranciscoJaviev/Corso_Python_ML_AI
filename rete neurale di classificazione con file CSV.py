import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd  # Aggiungi questa linea
import matplotlib.pyplot as plt # Mantieni l'import di matplotlib


# 1. Caricamento dei dati dal file SVC (CSV)
df = pd.read_csv('C:\\Users\\hp\\Documents\\Corso AI Cefi\\file .csv\\alzheimers_disease_data.csv')

#elimino tutte le colonne che non contengono numeri
df1 = df.drop(df.select_dtypes(include=['object']).columns, axis=1)


X = df1[['Age', 'AlcoholConsumption', 'PhysicalActivity']].values
y = df1[['Diagnosis']].values

# 2. Divisione dei Dati in Training e Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3. Pre-elaborazione dei Dati (Scaling)
# È buona pratica scalare le feature per aiutare la rete neurale a convergere più velocemente
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ... (caricamento dati come prima) ...

#num_classes = len(y.unique()) # Determina il numero di classi uniche

# 4. Definizione del Modello della Rete Neurale (Semplice)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu', input_shape=(X_train_scaled.shape[1],)), # Aggiorna input_shape
    tf.keras.layers.Dense(units=8, activation='relu'), # Aggiorna input_shape
    tf.keras.layers.Dense(units=1, activation='sigmoid') # Output binarios
])


# 5. Compilazione del Modello
model.compile(optimizer='adam',
              loss='binary_crossentropy', # Modifica la funzione di perdita
              metrics=['accuracy'])



# 6. Addestramento del Modello
epochs = 50                 # Numero di volte che l'intero set di training viene passato attraverso la rete
batch_size = 32             # Numero di campioni di training processati alla volta
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_scaled, y_test), verbose=0)

# 7. Valutazione del Modello sul Set di Test
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Perdita sul set di test: {loss:.4f}")
print(f"Accuratezza sul set di test: {accuracy:.4f}")

# 8. (Opzionale) Visualizzazione della Curva di Addestramento
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Accuratezza (Training)') #Indica come l'accuratezza migliora sui 
#dati di addestramento man mano che il modello si addestra

plt.plot(history.history['val_accuracy'], label='Accuratezza (Validazione)') #Indica l'accuratezza sui dati di 
#validazione, quindi su dati mai visti dal modello prima

plt.xlabel('Epoche') #Indica il numero di epoche, ovvero il numero di volte in cui il modello ha 
#visto il dataset completo durante l'addestramento

plt.ylabel('Accuratezza') #Mostra il livello di accuratezza del modello, cioè la frazione di esempi 
#classificati correttamente

plt.title('Curva di Addestramento')
plt.legend()
plt.grid(True)
plt.show()

# 9. (Opzionale) Fare Previsioni
predictions = model.predict(X_test_scaled)

# Le previsioni saranno array di probabilità per ogni classe.
print("\nPrime 10 Previsioni (Probabilità per Classe):")
print(predictions[:10])

# Per ottenere la classe predetta (l'indice con la probabilità più alta):
import numpy as np
binary_predict = (predictions > 0.5).astype(int)
print("\nPrime 10 Previsioni (Classe Predetta):")
print(binary_predict[:10].flatten())

print("\nPrime 10 Etichette Reali:")
print(y_test[:10])


'''
Possibili interpretazioni:
**Se entrambe le curve salgono e si stabilizzano a valori alti, il modello si sta addestrando correttamente.
**Se l'accuratezza di training continua a salire mentre quella di validazione cala, potrebbe esserci overfitting,
  cioè il modello impara troppo dai dati di training e non generalizza bene.
**Se entrambe le curve rimangono basse, il modello potrebbe essere sottopotenziato, cioè non abbastanza complesso
  per catturare i pattern nei dati.
'''