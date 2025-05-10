import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd  # Aggiungi questa linea
import matplotlib.pyplot as plt # Mantieni l'import di matplotlib


# 1. Caricamento dei dati dal file SVC (CSV)
df = pd.read_csv('C:\\Users\\hp\\Documents\\Corso AI Cefi\\file .csv\\Mall_Customers.csv')

#elimino tutte le colonne che non contengono numeri
df1 = df.drop(df.select_dtypes(include=['object']).columns, axis=1)


X = df1[['Age', 'Annual_Income_(k$)']].values
y = df1[['Spending_Score']].values

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
    tf.keras.layers.Dense(units=8, activation='relu', input_shape=(X_train_scaled.shape[1],)), # Aggiorna input_shape
    tf.keras.layers.Dense(units=4, activation='relu'), # Aggiorna input_shape
    tf.keras.layers.Dense(units=1,) 
])


# 5. Compilazione del Modello
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Cambia la funzione di perdita per la regressione
              metrics=['mae', 'mse'])  # - 'mae' sta per Mean Absolute Error, che misura la media del valore 
#assoluto delle differenze tra le previsioni e i valori reali. È meno sensibile agli outlier rispetto all'MSE.
# - 'mse' sta per Mean Squared Error, la stessa funzione di perdita. Includerla come metrica permette di 
#visualizzarne il valore durante l'addestramento.




# 6. Addestramento del Modello
epochs = 50                 # Numero di volte che l'intero set di training viene passato attraverso la rete
batch_size = 32             # Numero di campioni di training processati alla volta
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_scaled, y_test), verbose=0)

# 7. Valutazione del Modello sul Set di Test
loss, mse, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Perdita (MSE) sul set di test: {loss:.4f}")
print(f"Errore Quadratico Medio (MSE) sul set di test: {mse:.4f}")
print(f"Errore Assoluto Medio (MAE) sul set di test: {mae:.4f}")

# 8. (Opzionale) Visualizzazione della Curva di Addestramento
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perdita (Training)') # Questa riga traccia la curva della perdita (loss) sul set di training nel corso delle
# epoche.

plt.plot(history.history['val_loss'], label='Perdita (Validazione)') # Questa riga traccia la curva della perdita sul set di validazione 

plt.xlabel('Epoche') #Indica il numero di epoche, ovvero il numero di volte in cui il modello ha 
#visto il dataset completo durante l'addestramento

plt.ylabel('Perdita (MSE)') #Mostra il livello di accuratezza del modello, cioè la frazione di esempi 
#classificati correttamente

plt.title('Curva di Addestramento (Perdita)')
plt.legend()
plt.grid(True)
plt.show()


# Visualizzazione delle metriche aggiuntive (MAE)
plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='MAE (Training)')
plt.plot(history.history['val_mae'], label='MAE (Validazione)')
plt.xlabel('Epoche')
plt.ylabel('Errore Assoluto Medio (MAE)')
plt.title('Curva di Addestramento (MAE)')
plt.legend()
plt.grid(True)
plt.show()

# 9. (Opzionale) Fare Previsioni
predictions = model.predict(X_test_scaled)

# Le previsioni saranno array di probabilità per ogni classe.
print("\nPrime 10 Previsioni (Valori Continui):")
print(predictions[:10].flatten())  # Ora i valori predetti saranno numeri continui


print("\nPrime 10 Etichette Reali:")
print(y_test[:10].flatten())

'''
Curva di Perdita (Loss)

Perdita (Training): Questa linea ti mostra come l'errore del tuo modello è diminuito sul set di dati che ha utilizzato per imparare (il set di training) ad ogni epoca. Idealmente, vorresti vedere questa curva scendere costantemente nel tempo, indicando che il modello sta diventando più bravo a fare previsioni sui dati di training.
Perdita (Validazione): Questa linea, invece, ti dice come l'errore del tuo modello è cambiato su un set di dati separato che il modello non ha mai visto durante l'addestramento (il set di validazione). Questa è una metrica cruciale per capire se il tuo modello sta generalizzando bene a nuovi dati.
Interpretazione dell'Andamento della Curva di Perdita:

Andamento Ideale: Entrambe le curve (training e validazione) dovrebbero scendere e stabilizzarsi a un valore basso. La curva di validazione dovrebbe seguire da vicino la curva di training.
Overfitting: Se la curva di perdita sul training continua a diminuire significativamente, mentre la curva di perdita sulla validazione inizia a salire o si stabilizza a un valore più alto, questo è un segnale di overfitting. Significa che il modello sta imparando troppo bene i dettagli specifici del set di training (incluso il "rumore") e non riesce a generalizzare bene a nuovi dati.
Underfitting: Se entrambe le curve di perdita rimangono relativamente alte e non mostrano un calo significativo, potrebbe indicare underfitting. In questo caso, il modello potrebbe non avere abbastanza capacità o non essere stato addestrato per un numero sufficiente di epoche per catturare le relazioni importanti nei dati.
Buona Generalizzazione: Se entrambe le curve diminuiscono in modo simile e si stabilizzano a un valore ragionevolmente basso, il modello sta probabilmente generalizzando bene ai nuovi dati.
Instabilità/Rumore: Oscillazioni significative nelle curve, specialmente nella curva di validazione, potrebbero indicare che il processo di addestramento è instabile o che i dati di validazione sono troppo piccoli o rumorosi.
Curva di Errore Assoluto Medio (MAE)

MAE (Training): Simile alla perdita di training, questa curva mostra come l'errore assoluto medio sul set di training è cambiato nel tempo. L'MAE è un'altra metrica per valutare le prestazioni del modello, rappresentando la media del valore assoluto degli errori tra le previsioni del modello e i valori reali.
MAE (Validazione): Questa curva mostra l'errore assoluto medio sul set di validazione ad ogni epoca.
Interpretazione dell'Andamento della Curva MAE:

L'interpretazione delle curve MAE è molto simile a quella delle curve di perdita:

Andamento Ideale: Entrambe le curve dovrebbero scendere e stabilizzarsi a un valore basso.
Overfitting: Se l'MAE sul training continua a diminuire mentre l'MAE sulla validazione inizia a salire o si stabilizza a un valore più alto.
Underfitting: Se entrambe le curve MAE rimangono relativamente alte.
Buona Generalizzazione: Se entrambe le curve diminuiscono in modo simile e si stabilizzano a un valore ragionevolmente basso.

'''