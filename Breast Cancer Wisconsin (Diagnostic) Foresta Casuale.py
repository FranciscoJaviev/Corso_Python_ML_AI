'''Nome Script:  Analisi sul cancro al seno tramite foresta casuale.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di classificare
il tipo di tumore di alcune soggetti.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [2 maggio 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



df = pd.read_csv('Breast Cancer Wisconsin (Diagnostic).csv')

X = df[["radius_mean","texture_mean","perimeter_mean","area_mean"]]
y = df['diagnosis']
class_name = ['Maligno', 'Benigno'] # Assumo che 0 = Sano, 1 = Malato
#class_name = list(y['Diagnosis'].unique()) 
feature_names=["radius_mean","texture_mean","perimeter_mean","area_mean"]
#lunghezza_names = max(len(cn) for cn in class_name)  # Ora sono tutte stringhe

# 2. Divisione dei Dati in Training e Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Creazione e Addestramento del Modello della Foresta Casuale
# Puoi sperimentare con diversi numeri di alberi (n_estimators) e altri parametri
foresta_casuale = RandomForestClassifier(n_estimators=100, random_state=42)
foresta_casuale.fit(X_train, y_train)

# 4. Previsioni sul Set di Test
y_pred = foresta_casuale.predict(X_test)

# 5. Valutazione del Modello
print("Risultati della Foresta Casuale:")
print("\nMatrice di Confusione:")

# Crea una figura con due subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 7))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_name, yticklabels=class_name, ax=axs[0]) #ax=axs[0] mi specifica l'asse


axs[0].set_ylabel('Valori Reali')
axs[0].set_xlabel('Valori Predetti')
axs[0].set_title('Matrice di Confusione (Foresta Casuale)')
#plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_name))

# 6. (Opzionale) Stima dell'Importanza delle Feature (varibili predittive)
feature_importances = foresta_casuale.feature_importances_
#feature_names = feature_names
#sorted_indices = feature_importances.argsort()[::-1]
sorted_indices = np.argsort(feature_importances)[::-1]  # Assicura corretto ordinamento per importanza delle etichette


axs[1].set_title("Importanza delle Feature nella Foresta Casuale")
axs[1].bar(range(X.shape[1]), feature_importances[sorted_indices], align="center")
axs[1].set_xticks(range(X.shape[1]), [feature_names[i] for i in sorted_indices], rotation=45)
axs[1].set_xlabel("Feature")
axs[1].set_ylabel("Punteggio di importanza")
plt.tight_layout()
plt.show()

