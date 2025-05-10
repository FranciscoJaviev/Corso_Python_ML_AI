'''Nome Script:  Analisi sulla alzaimer negli adulti tramite la foresta casuale.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di predire 
l'insorgenza dell'alzaimer nei soggetti adulti.

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



df = pd.read_csv('alzheimers_disease_data.csv')

X = df[['Age','Gender','Ethnicity','EducationLevel']]
y = df['Diagnosis']
class_name = ['Sano', 'Malato'] # Assumo che 0 = Sano, 1 = Malato
feature_names=['Age', 'Gender', 'Ethnicity', 'EducationLevel']

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
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_name, yticklabels=class_name)
plt.ylabel('Valori Reali')
plt.xlabel('Valori Predetti')
plt.title('Matrice di Confusione (Foresta Casuale)')
#plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_name))

# 6. (Opzionale) Stima dell'Importanza delle Feature
feature_importances = foresta_casuale.feature_importances_
#feature_names = feature_names
#sorted_indices = feature_importances.argsort()[::-1]
sorted_indices = np.argsort(feature_importances)[::-1]  # Assicura corretto ordinamento

plt.figure(figsize=(8, 6))
plt.title("Importanza delle Feature nella Foresta Casuale")
plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in sorted_indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importanza")
plt.tight_layout()
plt.show()
