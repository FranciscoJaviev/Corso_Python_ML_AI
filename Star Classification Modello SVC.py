'''Nome Script:  Analisi tipo di corpo celesti tramite i dati acquisiti utilizzando il modello SVC.py
Descrizione: Questo script elabora e analizza i dati di di un database e cerca di classificare a quele tipologia
di corpo celeste appartiene.

Autore: [Francisco J. Scognamiglio]
Versione: 1.0
Data: [4 maggio 2025]
Copyright: © [Francisco J. Scognamiglio] [2025]
Licenza: [MIT]
'''


import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('star_classification.csv')

X = df[["alpha","delta","u","g","r","i","z"]]
y = df['class']

class_name = ['Galassia', 'Stella', 'QSO']  
feature_names=["alpha","delta","u","g","r","i","z"]



#Divisione dei Dati in Training e Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Creazione e Addestramento del Modello SVC
#Puoi sperimentare con diversi kernel ('linear', 'poly', 'rbf', 'sigmoid') e parametri
modello_svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=True)
modello_svc.fit(X_train, y_train)

#Previsioni sul Set di Test
y_pred = modello_svc.predict(X_test)

#Valutazione del Modello
print("\nRisultati della Classificazione SVC:")
print("\nMatrice di Confusione:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_name, yticklabels=class_name)
plt.ylabel('Valori Reali')
plt.xlabel('Valori Predetti')
plt.title('Matrice di Confusione (SVC)')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_name, zero_division=1))
